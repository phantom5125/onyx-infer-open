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

using onyx::Tensor;
using Clock = std::chrono::steady_clock;
constexpr int64_t kHidden = 5376;
constexpr std::array<uint32_t, 2> kGates{2, 5};

Tensor make_bf16(id<MTLDevice> device, std::vector<int64_t> shape, uint32_t seed, uint64_t offset = 0) {
  Tensor tensor{nil, std::move(shape), MPSDataTypeBFloat16, offset};
  tensor.buffer = [device newBufferWithLength:offset + tensor.byte_size() options:MTLResourceStorageModeShared];
  if (!tensor.buffer) throw std::runtime_error("Could not allocate BF16 probe tensor");
  auto* values = static_cast<uint16_t*>(tensor.data());
  for (uint64_t i = 0; i < tensor.elements(); ++i) {
    seed = seed * 1664525U + 1013904223U;
    const float value = (static_cast<int32_t>(seed >> 16) - 32768) / 16384.0f;
    values[i] = onyx::float_to_bf16(value);
  }
  return tensor;
}

void require_equal(const Tensor& reference, const Tensor& actual, const char* comparison) {
  if (std::memcmp(reference.data(), actual.data(), reference.byte_size()) == 0) return;
  const auto* expected = static_cast<const uint16_t*>(reference.data());
  const auto* received = static_cast<const uint16_t*>(actual.data());
  for (uint64_t i = 0; i < reference.elements(); ++i) {
    if (expected[i] == received[i]) continue;
    std::fprintf(stderr, "%s at cell %llu: expected %04x, received %04x\n",
                 comparison, i, expected[i], received[i]);
    break;
  }
  throw std::runtime_error("DiT gate probe failed bitwise check");
}

void update_cpu(Tensor& residual, const Tensor& value, const Tensor& table, const Tensor& indices,
                uint32_t gate_part) {
  auto* result = static_cast<uint16_t*>(residual.data());
  const auto* updates = static_cast<const uint16_t*>(value.data());
  const auto* gates = static_cast<const uint16_t*>(table.data());
  const auto* modalities = static_cast<const int32_t*>(indices.data());
  for (int64_t row = 0; row < residual.shape[0]; ++row) {
    const auto* gate = gates + (modalities[row] * 6 + gate_part) * kHidden;
    for (int64_t column = 0; column < kHidden; ++column) {
      const auto cell = row * kHidden + column;
      const float updated = std::fma(onyx::bf16_to_float(updates[cell]), onyx::bf16_to_float(gate[column]),
                                     onyx::bf16_to_float(result[cell]));
      result[cell] = onyx::float_to_bf16(updated);
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  @autoreleasepool {
    try {
      if (argc < 2) {
        throw std::runtime_error("Usage: onyx-dit-gate-probe ROWS [--verify-only] [--references]");
      }
      const int64_t rows = std::stoll(argv[1]);
      bool verify_only = false, references = false;
      for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (option == "--verify-only") verify_only = true;
        else if (option == "--references") references = true;
        else throw std::runtime_error("Invalid gate probe option");
      }
      if (rows < 1) throw std::runtime_error("Gate probe row count must be positive");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("Metal device unavailable");
      onyx::GraphSession session(device);
      Tensor baseline = make_bf16(device, {rows, kHidden}, 1);
      Tensor metal = make_bf16(device, {rows, kHidden}, 1);
      Tensor cpu = make_bf16(device, {rows, kHidden}, 1);
      const Tensor value = make_bf16(device, {rows, kHidden}, 2);
      // The DiT table is a view into a reusable cache buffer.
      const Tensor table = make_bf16(device, {references ? 18 : 12, kHidden}, 3, 256);
      Tensor indices{nil, {rows}, MPSDataTypeInt32, 64};
      indices.buffer = [device newBufferWithLength:indices.byte_offset + indices.byte_size()
                                          options:MTLResourceStorageModeShared];
      if (!indices.buffer) throw std::runtime_error("Could not allocate gate modality indices");
      auto* tags = static_cast<int32_t*>(indices.data());
      if (references) {
        for (int64_t row = 0; row < rows; ++row) tags[row] = (row + 2) % 3;
      } else {
        const int64_t text = std::min<int64_t>(rows, 69);
        std::fill(tags, tags + text, 1);
        std::fill(tags + text, tags + rows, 0);
      }

      // The same baseline checks the indexed norms and the in-place gate.
      std::array<Tensor, 2> expanded;
      @autoreleasepool {
        constexpr double epsilon = 1e-5;
        const auto parts = session.modulation_parts(table, indices);
        const Tensor weight = make_bf16(device, {kHidden}, 4);
        for (size_t i = 0; i < kGates.size(); ++i) {
          const uint32_t gate = kGates[i];
          const uint32_t shift = gate - 2;
          const Tensor norm = session.modulated_rms_norm(baseline, weight, parts[shift + 1], parts[shift], epsilon);
          const Tensor indexed_norm = session.indexed_modulated_rms_norm(metal, weight, table, indices, shift, epsilon);
          require_equal(norm, indexed_norm, "Expanded vs. indexed modulation norm");

          expanded[i] = parts[gate];
          const Tensor indexed_gate = session.indexed_gated_residual(baseline, value, table, indices, gate);
          baseline = session.gated_residual(baseline, value, expanded[i]);
          session.metal_gated_residual(metal, value, table, indices, gate);
          update_cpu(cpu, value, table, indices, gate);
          require_equal(baseline, indexed_gate, "Expanded vs. indexed MPSGraph gate");
          require_equal(cpu, metal, "CPU vs. Metal gate");
          require_equal(baseline, metal, "Expanded MPSGraph vs. Metal gate");
        }
      }

      const auto step = [&](bool use_metal, size_t gate) {
        @autoreleasepool {
          if (use_metal) {
            session.metal_gated_residual(metal, value, table, indices, kGates[gate]);
          } else {
            baseline = session.gated_residual(baseline, value, expanded[gate]);
          }
        }
      };
      std::printf("{\"rows\":%lld,\"width\":%lld,\"table_offset\":%llu,\"bitwise_equal\":true,"
                  "\"cpu_bitwise_equal\":true,\"timings_ms\":[", rows, kHidden, table.byte_offset);
      if (!verify_only) {
        for (size_t warmup = 0; warmup < 16; ++warmup) {
          step(false, warmup % 2);
          step(true, warmup % 2);
        }
        constexpr std::array<bool, 4> order{false, true, true, false};
        for (size_t block = 0; block < 3; ++block) {
          std::printf("%s[", block ? "," : "");
          for (size_t repeat = 0; repeat < 12; ++repeat) {
            for (size_t slot = 0; slot < order.size(); ++slot) {
              const bool use_metal = order[slot] != (block == 1);
              const auto start = Clock::now();
              step(use_metal, repeat % 2);
              const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
              const char* variant = use_metal ? "metal" : "mpsgraph";
              std::printf("%s{\"variant\":\"%s\",\"ms\":%.6f}", (repeat || slot) ? "," : "",
                          variant, elapsed);
            }
          }
          std::printf("]");
        }
        require_equal(baseline, metal, "MPSGraph vs. Metal after repeated updates");
      }
      std::printf("]}\n");
      return 0;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
}
