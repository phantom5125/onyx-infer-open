#import "native.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace {

constexpr int64_t kHidden = 5376;
constexpr int64_t kReference16Rows = 33942;
using Clock = std::chrono::steady_clock;
using onyx::Tensor;

Tensor bf16_tensor(id<MTLDevice> device, std::vector<int64_t> shape, uint32_t seed) {
  Tensor tensor{nil, std::move(shape), MPSDataTypeBFloat16};
  tensor.buffer = [device newBufferWithLength:tensor.byte_size() options:MTLResourceStorageModeShared];
  if (!tensor.buffer) throw std::runtime_error("Could not allocate probe tensor");
  auto* values = static_cast<uint16_t*>(tensor.data());
  uint32_t state = seed;
  for (uint64_t i = 0; i < tensor.elements(); ++i) {
    state = state * 1664525U + 1013904223U;
    values[i] = onyx::float_to_bf16((static_cast<int32_t>(state >> 16) - 32768) / 32768.0f);
  }
  return tensor;
}

template <typename Apply>
void print_timings(const Apply& apply) {
  // Settle the allocator and GPU clock before starting the alternating samples.
  for (uint32_t warmup = 0; warmup < 12; ++warmup) {
    @autoreleasepool {
      apply(false);
      apply(true);
    }
  }
  constexpr uint32_t repeats = 10;
  for (uint32_t block = 0; block < 3; ++block) {
    for (uint32_t warmup = 0; warmup < 3; ++warmup) {
      @autoreleasepool {
        apply(false);
        apply(true);
      }
    }
    std::printf("%s[", block ? "," : "");
    const std::array<bool, 4> order = {false, true, true, false};
    for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
      for (size_t i = 0; i < order.size(); ++i) {
        // Rotate ABBA to BAAB in the middle block to balance position.
        const bool indexed = order[i] != (block == 1);
        const auto start = Clock::now();
        @autoreleasepool { apply(indexed); }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        std::printf("%s{\"variant\":\"%s\",\"ms\":%.6f}", (repeat || i) ? "," : "",
                    indexed ? "indexed" : "baseline", ms);
      }
    }
    std::printf("]");
  }
}

void verify_text_modulation(id<MTLDevice> device, onyx::GraphSession& session, int64_t rows,
                            bool verify_only) {
  if (rows < 3) throw std::runtime_error("Text modulation needs at least three tokens");
  const auto x = bf16_tensor(device, {rows, kHidden}, 1);
  const auto attended = bf16_tensor(device, {rows, kHidden}, 2);
  const auto mlp = bf16_tensor(device, {rows, kHidden}, 3);
  const auto norm1 = bf16_tensor(device, {kHidden}, 4);
  const auto norm2 = bf16_tensor(device, {kHidden}, 5);
  constexpr int64_t table_rows = 12;
  const std::array<Tensor, 2> contiguous{
      bf16_tensor(device, {table_rows, kHidden}, 6),
      bf16_tensor(device, {table_rows, kHidden}, 7)};
  const Tensor timesteps = bf16_tensor(device, {2, table_rows, kHidden}, 8);
  auto* backing = static_cast<uint16_t*>(timesteps.data());
  for (const auto& step : contiguous) {
    std::memcpy(backing, step.data(), step.byte_size());
    backing += step.elements();
  }
  const auto view = [&](size_t step) {
    Tensor table = timesteps;
    table.shape = contiguous[step].shape;
    table.byte_offset += step * table.byte_size();
    return table;
  };
  Tensor indices{nil, {rows}, MPSDataTypeInt32};
  indices.buffer = [device newBufferWithLength:indices.byte_size() options:MTLResourceStorageModeShared];
  if (!indices.buffer) throw std::runtime_error("Could not allocate modality indices");
  auto* tags = static_cast<int32_t*>(indices.data());

  const auto apply = [&](const Tensor& table, bool indexed) -> std::array<Tensor, 6> {
    constexpr double epsilon = 1e-5;
    const auto parts = indexed ? session.modulation_gates(table, indices)
                               : session.modulation_parts(table, indices);
    const auto& gate1 = parts[indexed ? 0 : 2];
    const auto& gate2 = parts[indexed ? 1 : 5];
    const auto a = indexed ? session.indexed_modulated_rms_norm(x, norm1, table, indices, 0, epsilon)
                           : session.modulated_rms_norm(x, norm1, parts[1], parts[0], epsilon);
    const auto r = session.gated_residual(x, attended, gate1);
    const auto b = indexed ? session.indexed_modulated_rms_norm(r, norm2, table, indices, 3, epsilon)
                           : session.modulated_rms_norm(r, norm2, parts[4], parts[3], epsilon);
    const auto y = session.gated_residual(r, mlp, gate2);
    return {a, b, gate1, gate2, r, y};
  };
  const std::array<size_t, 3> steps{0, 1, 0};
  const std::array<int64_t, 3> splits{rows / 3, 2 * rows / 3, rows / 3};
  const std::array<const char*, 6> outputs{"norm1", "norm2", "gate2", "gate5", "residual1", "residual2"};
  for (size_t run = 0; run < steps.size(); ++run) {
    @autoreleasepool {
      const auto step = steps[run];
      std::fill(tags, tags + splits[run], 1);
      std::fill(tags + splits[run], tags + rows, 0);
      const auto expected = apply(contiguous[step], false);
      const auto baseline = apply(view(step), false);
      const auto candidate = apply(view(step), true);
      for (size_t part = 0; part < outputs.size(); ++part) {
        const auto& reference = expected[part];
        if (std::memcmp(reference.data(), baseline[part].data(), reference.byte_size()) ||
            std::memcmp(reference.data(), candidate[part].data(), reference.byte_size())) {
          throw std::runtime_error(std::string("Text modulation differs at ") + outputs[part] +
                                   ", run " + std::to_string(run) + ", step " + std::to_string(step));
        }
      }
    }
  }
  const auto expanded_bytes = static_cast<unsigned long long>(x.byte_size());
  std::printf("{\"rows\":%lld,\"text_only\":true,\"bitwise_equal\":true,\"steps\":[0,1,0],"
              "\"text_tokens\":[%lld,%lld,%lld],\"baseline_modulation_output_bytes\":%llu,"
              "\"candidate_modulation_output_bytes\":%llu,\"timings_ms\":[",
              static_cast<long long>(rows), static_cast<long long>(splits[0]),
              static_cast<long long>(splits[1]), static_cast<long long>(splits[2]),
              6 * expanded_bytes, 2 * expanded_bytes);
  if (!verify_only) {
    const auto table = view(1);
    const int64_t text_tokens = std::min<int64_t>(69, rows - 1);
    std::fill(tags, tags + text_tokens, 1);
    std::fill(tags + text_tokens, tags + rows, 0);
    print_timings([&](bool indexed) { return apply(table, indexed); });
  }
  std::printf("]}\n");
}

}  // namespace

int main(int argc, char* argv[]) {
  @autoreleasepool {
    try {
      if (argc < 2 || argc > 4) {
        throw std::runtime_error("Usage: onyx-dit-modulation-probe ROWS [--verify-only] [--text-only]");
      }
      const int64_t rows = std::stoll(argv[1]);
      bool verify_only = false;
      bool text_only = false;
      for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--verify-only" && !verify_only) verify_only = true;
        else if (option == "--text-only" && !text_only) text_only = true;
        else throw std::runtime_error("Invalid probe options");
      }
      if (rows < 1) throw std::runtime_error("Probe row count must be positive");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("Metal device unavailable");
      onyx::GraphSession session(device);
      if (text_only) {
        verify_text_modulation(device, session, rows, verify_only);
        return 0;
      }
      const auto x = bf16_tensor(device, {rows, kHidden}, 1);
      const auto attended = bf16_tensor(device, {rows, kHidden}, 2);
      const auto mlp = bf16_tensor(device, {rows, kHidden}, 3);
      const auto norm1 = bf16_tensor(device, {kHidden}, 4);
      const auto norm2 = bf16_tensor(device, {kHidden}, 5);
      const auto table = bf16_tensor(device, {18, kHidden}, 6);
      Tensor indices{nil, {rows}, MPSDataTypeInt32};
      indices.buffer = [device newBufferWithLength:indices.byte_size() options:MTLResourceStorageModeShared];
      if (!indices.buffer) throw std::runtime_error("Could not allocate modality indices");
      auto* tags = static_cast<int32_t*>(indices.data());
      if (rows == kReference16Rows) {
        constexpr int64_t square_patches = 32 * 32;
        auto* tag = tags;
        for (int image = 0; image < 16; ++image) {
          tag = std::fill_n(tag, image < 9 ? 5 : 6, 1);
          tag = std::fill_n(tag, square_patches + 2, 0);
        }
        tag = std::fill_n(tag, 31, 1);
        tag = std::fill_n(tag, 16 * square_patches, 2);
        std::fill_n(tag, square_patches, 0);
      } else {
        for (int64_t row = 0; row < rows; ++row) tags[row] = (row + 2) % 3;
      }

      const auto apply = [&](bool indexed) -> std::array<Tensor, 4> {
        constexpr double epsilon = 1e-5;
        if (indexed) {
          auto a = session.indexed_modulated_rms_norm(x, norm1, table, indices, 0, epsilon);
          auto r = session.indexed_gated_residual(x, attended, table, indices, 2);
          auto b = session.indexed_modulated_rms_norm(r, norm2, table, indices, 3, epsilon);
          auto y = session.indexed_gated_residual(r, mlp, table, indices, 5);
          return {a, r, b, y};
        }
        const auto parts = session.modulation_parts(table, indices);
        auto a = session.modulated_rms_norm(x, norm1, parts[1], parts[0], epsilon);
        auto r = session.gated_residual(x, attended, parts[2]);
        auto b = session.modulated_rms_norm(r, norm2, parts[4], parts[3], epsilon);
        auto y = session.gated_residual(r, mlp, parts[5]);
        return {a, r, b, y};
      };

      uint64_t unequal_values = 0;
      float max_abs_error = 0;
      @autoreleasepool {
        const auto baseline = apply(false);
        const auto candidate = apply(true);
        for (size_t i = 0; i < baseline.size(); ++i) {
          const auto* left = static_cast<const uint16_t*>(baseline[i].data());
          const auto* right = static_cast<const uint16_t*>(candidate[i].data());
          if (std::memcmp(left, right, baseline[i].byte_size()) == 0) continue;
          for (uint64_t cell = 0; cell < baseline[i].elements(); ++cell) {
            if (left[cell] == right[cell]) continue;
            ++unequal_values;
            max_abs_error = std::max(max_abs_error, std::abs(onyx::bf16_to_float(left[cell]) -
                                                            onyx::bf16_to_float(right[cell])));
          }
        }
      }
      const bool equal = unequal_values == 0;
      std::printf("{\"rows\":%lld,\"bitwise_equal\":%s,\"unequal_values\":%llu,\"max_abs_error\":%g,"
                  "\"baseline_modulation_output_bytes\":%llu,\"candidate_modulation_output_bytes\":0,\"timings_ms\":[",
                  static_cast<long long>(rows), equal ? "true" : "false",
                  static_cast<unsigned long long>(unequal_values), max_abs_error,
                  static_cast<unsigned long long>(6 * x.byte_size()));
      if (equal && !verify_only) print_timings(apply);
      std::printf("]}\n");
      return equal ? 0 : 1;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
}
