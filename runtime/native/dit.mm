#import "native.hpp"
#import <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string_view>
#include <vector>

namespace onyx {
namespace {

constexpr uint32_t kDiTBlocks = 50;
// Weights one DiT block consumes, in the order the block reads them.
const std::vector<std::string> kBlockTensors = {
    "norm1.weight", "attn.qkv_proj.weight", "attn.q_norm.weight", "attn.k_norm.weight",
    "attn.out_proj.weight", "norm2.weight", "mlp.fc1.weight", "mlp.fc2.weight"};
// Weight and bias appended to a linear-layer name, such as an AdaLN projection.
const std::vector<std::string> kLinearSuffixes = {".weight", ".bias"};
constexpr float kReferenceSigma = 0.001f;
constexpr int64_t kSamplingSteps = 6;
constexpr int64_t kHiddenSize = 5376;

std::vector<int64_t> modulation_shape(int64_t timesteps, bool final) {
  return {timesteps, final ? 1 : 3, final ? 2 : 6, kHiddenSize};
}

struct ReferenceConditioning {
  using Grid = std::pair<uint32_t, uint32_t>;
  std::vector<Grid> grids;
  Tensor text_tags{};
  Tensor patches{};
};

ReferenceConditioning load_references(const std::string& path, int64_t text_count) {
  if (path.empty()) return {};
  WeightStore store(path);
  ReferenceConditioning references;
  const Tensor grids = store.tensor("grids");
  if (grids.dtype != MPSDataTypeInt32 || grids.shape.size() != 2 ||
      grids.shape[1] != 2 || grids.shape[0] < 1 || grids.shape[0] > 16) {
    throw std::runtime_error("Invalid H3 reference grids");
  }
  const auto* data = static_cast<const int32_t*>(grids.data());
  int64_t patch_count = 0;
  for (int64_t image = 0; image < grids.shape[0]; ++image) {
    const int32_t height = data[image * 2];
    const int32_t width = data[image * 2 + 1];
    if (height < 1 || width < 1 || height > 256 || width > 256) {
      throw std::runtime_error("Invalid H3 reference patch dimensions");
    }
    references.grids.emplace_back(height, width);
    patch_count += height * width;
  }
  references.patches = store.tensor("patches");
  references.text_tags = store.tensor("text_tags");
  if (references.patches.dtype != MPSDataTypeFloat32 ||
      references.patches.shape != std::vector<int64_t>{patch_count, 96} ||
      references.text_tags.dtype != MPSDataTypeInt32 ||
      references.text_tags.shape != std::vector<int64_t>{text_count}) {
    throw std::runtime_error("Invalid H3 reference conditioning tensors");
  }
  const auto* tags = static_cast<const int32_t*>(references.text_tags.data());
  if (std::any_of(tags, tags + text_count, [](int32_t tag) { return tag != 0 && tag != 1; })) {
    throw std::runtime_error("Invalid H3 text modality tag");
  }
  return references;
}

Tensor make_tensor(const std::vector<int64_t>& shape, MPSDataType dtype, const std::vector<float>& values) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (dtype == MPSDataTypeBFloat16) {
    std::vector<uint16_t> data;
    data.reserve(values.size());
    for (float value : values) data.push_back(float_to_bf16(value));
    id<MTLBuffer> buffer = [device newBufferWithBytes:data.data() length:data.size() * 2 options:MTLResourceStorageModeShared];
    return Tensor{buffer, shape, dtype};
  }
  id<MTLBuffer> buffer = [device newBufferWithBytes:values.data() length:values.size() * sizeof(float) options:MTLResourceStorageModeShared];
  return Tensor{buffer, shape, dtype};
}

Tensor scalar(float value, MPSDataType dtype) {
  return make_tensor({1}, dtype, {value});
}

std::vector<float> read_sigmas(const std::string& model) {
  const auto path = std::filesystem::path(model).parent_path() / "pdd_config.json";
  @autoreleasepool {
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
    if (!data) throw std::runtime_error("Missing PDD schedule: " + path.string());
    NSError* error = nil;
    NSDictionary* root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (!root || error) throw std::runtime_error("Invalid PDD schedule");
    std::vector<float> result;
    for (NSNumber* value in root[@"sigmas"]) result.push_back(value.floatValue);
    if (result.size() != 7) throw std::runtime_error("Expected six PDD steps");
    return result;
  }
}

Tensor timestep_projection(const std::vector<float>& sigmas) {
  std::vector<float> values(sigmas.size() * 256);
  for (size_t step = 0; step < sigmas.size(); ++step) {
    const float timestep = 1.0f - sigmas[step];
    for (uint32_t index = 0; index < 128; ++index) {
      const double angle = timestep * std::exp(-std::log(10000.0) * index / 128.0);
      values[step * 256 + index] = std::cos(angle);
      values[step * 256 + 128 + index] = std::sin(angle);
    }
  }
  return make_tensor({static_cast<int64_t>(sigmas.size()), 256}, MPSDataTypeFloat32, values);
}

Tensor original_projection(GraphSession& session, WeightStore& store, const std::string& prefix, const Tensor& temb, bool final) {
  Tensor activated = session.silu(temb);
  activated = session.cast(activated, MPSDataTypeBFloat16);
  Tensor value = session.linear(store, prefix + ".adaln_proj.linear", activated);
  return session.reshape(value, modulation_shape(temb.shape[0], final));
}

Tensor time_embedding(GraphSession& session, WeightStore& store, const std::vector<float>& sigmas, bool prefetch) {
  Tensor temb = timestep_projection(sigmas);
  if (prefetch) store.prefetch("time_embedder.", {"proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias"});
  temb = session.linear(store, "time_embedder.proj_in", temb);
  temb = session.silu(temb);
  return session.linear(store, "time_embedder.proj_out", temb);
}

// The AdaLN table of block `layer`, or of the final layer when layer == kDiTBlocks.
Tensor modulation_table(GraphSession& session, WeightStore& store, uint32_t layer, const Tensor& temb) {
  return layer < kDiTBlocks
      ? original_projection(session, store, "blocks." + std::to_string(layer), temb, false)
      : original_projection(session, store, "final_layer", temb, true);
}

// Queue the weights of table `layer`. Nothing is queued past the final layer.
void queue_modulation_table(WeightStore& store, uint32_t layer) {
  if (layer > kDiTBlocks) return;
  store.prefetch(layer < kDiTBlocks ? "blocks." + std::to_string(layer) + ".adaln_proj.linear"
                                    : "final_layer.adaln_proj.linear",
                 kLinearSuffixes);
}

std::vector<Tensor> compute_modulations(GraphSession& session, WeightStore& store, const std::vector<float>& sigmas,
                                        bool prefetch) {
  const Tensor temb = time_embedding(session, store, sigmas, prefetch);
  // Each projection streams a 520 MB BF16 weight. Two stay in flight so that the reads
  // overlap instead of pacing the projections.
  if (prefetch) {
    queue_modulation_table(store, 0);
    queue_modulation_table(store, 1);
  }
  std::vector<Tensor> modulations;
  for (uint32_t layer = 0; layer <= kDiTBlocks; ++layer) {
    @autoreleasepool {
      if (prefetch) queue_modulation_table(store, layer + 2);
      modulations.push_back(modulation_table(session, store, layer, temb));
    }
  }
  return modulations;
}

using Digest = std::array<uint8_t, CC_SHA256_DIGEST_LENGTH>;

Digest sha256(const void* bytes, size_t length) {
  Digest digest;
  CC_SHA256(bytes, static_cast<CC_LONG>(length), digest.data());
  return digest;
}

bool modulation_weights_are_raw(const WeightStore& store) {
  for (const auto& name : store.names()) {
    if ((name.starts_with("time_embedder.") || name.find(".adaln_proj.") != std::string::npos) &&
        store.dtype(name).starts_with("Q4")) {
      return false;
    }
  }
  return true;
}

Digest modulation_cache_key(id<MTLDevice> device, const std::string& weights, const std::vector<float>& sigmas,
                            NSData* binary) {
  const auto build = sha256(binary.bytes, binary.length);
  NSData* device_and_os = [[NSString stringWithFormat:@"%@\n%@",
      device.name, NSProcessInfo.processInfo.operatingSystemVersionString] dataUsingEncoding:NSUTF8StringEncoding];
  NSMutableData* key = [NSMutableData dataWithBytes:build.data() length:build.size()];
  [key appendBytes:sigmas.data() length:sigmas.size() * sizeof(float)];
  [key appendBytes:weights.c_str() length:weights.size() + 1];
  [key appendData:device_and_os];
  return sha256(key.bytes, key.length);
}

NSString* modulation_cache_path(NSString* directory, const Digest& key) {
  constexpr char hex[] = "0123456789abcdef";
  std::string name;
  for (uint8_t byte : key) {
    name += hex[byte >> 4];
    name += hex[byte & 15];
  }
  name += ".adaln";
  return [directory stringByAppendingPathComponent:[NSString stringWithUTF8String:name.c_str()]];
}

std::vector<Tensor> load_modulations(id<MTLDevice> device, NSString* path, const Digest& key, int64_t timesteps) {
  const auto block_shape = modulation_shape(timesteps, false);
  const auto final_shape = modulation_shape(timesteps, true);
  const size_t block_bytes = timesteps * 3 * 6 * kHiddenSize * sizeof(uint16_t);
  const size_t payload_bytes = kDiTBlocks * block_bytes + timesteps * 2 * kHiddenSize * sizeof(uint16_t);
  NSData* data = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:nil];
  if (data.length != 2 * key.size() + payload_bytes) return {};
  const auto* bytes = static_cast<const uint8_t*>(data.bytes);
  if (std::memcmp(bytes, key.data(), key.size()) != 0) return {};
  const auto* payload = bytes + 2 * key.size();
  const auto checksum = sha256(payload, payload_bytes);
  if (std::memcmp(bytes + key.size(), checksum.data(), checksum.size()) != 0) return {};

  id<MTLBuffer> buffer = [device newBufferWithBytes:payload length:payload_bytes options:MTLResourceStorageModeShared];
  if (!buffer) return {};
  std::vector<Tensor> result;
  for (uint32_t layer = 0; layer < kDiTBlocks; ++layer) {
    result.push_back({buffer, block_shape, MPSDataTypeBFloat16, layer * block_bytes});
  }
  result.push_back({buffer, final_shape, MPSDataTypeBFloat16, kDiTBlocks * block_bytes});
  return result;
}

bool save_modulations(NSString* directory, NSString* path, const Digest& key, const std::vector<Tensor>& modulations) {
  if (![NSFileManager.defaultManager createDirectoryAtPath:directory withIntermediateDirectories:YES
                                               attributes:@{NSFilePosixPermissions: @0700} error:nil]) return false;
  // The cache format stores the key and payload digest before the packed BF16 tables.
  size_t payload_bytes = 0;
  for (const Tensor& tensor : modulations) payload_bytes += tensor.byte_size();
  NSMutableData* data = [NSMutableData dataWithLength:2 * key.size() + payload_bytes];
  auto* bytes = static_cast<uint8_t*>(data.mutableBytes);
  std::memcpy(bytes, key.data(), key.size());
  size_t offset = 2 * key.size();
  for (const Tensor& tensor : modulations) {
    std::memcpy(bytes + offset, tensor.data(), tensor.byte_size());
    offset += tensor.byte_size();
  }
  const auto checksum = sha256(bytes + 2 * key.size(), payload_bytes);
  std::memcpy(bytes + key.size(), checksum.data(), checksum.size());
  return [data writeToFile:path options:NSDataWritingAtomic error:nil];
}

std::vector<Tensor> get_modulations(GraphSession& session, WeightStore& store, const std::vector<float>& sigmas,
                                   id<MTLDevice> device, const std::string& cache_dir, bool prefetch, const char*& cache_status) {
  if (cache_dir.empty()) {
    cache_status = "disabled";
    return compute_modulations(session, store, sigmas, prefetch);
  }
  @autoreleasepool {
    NSString* directory = [NSString stringWithUTF8String:cache_dir.c_str()];
    NSString* binary_path = NSBundle.mainBundle.executablePath;
    NSData* binary = binary_path ? [NSData dataWithContentsOfFile:binary_path options:NSDataReadingMappedIfSafe error:nil] : nil;
    const std::string weights = store.file_identity();
    if (!directory || !binary.length || binary.length > UINT32_MAX || weights.empty() || !modulation_weights_are_raw(store)) {
    cache_status = "unavailable";
    return compute_modulations(session, store, sigmas, prefetch);
  }

    const auto key = modulation_cache_key(device, weights, sigmas, binary);
    NSString* path = modulation_cache_path(directory, key);
    auto result = load_modulations(device, path, key, sigmas.size());
    if (!result.empty()) {
      cache_status = "hit";
      return result;
    }

    result = compute_modulations(session, store, sigmas, prefetch);
    cache_status = weights == store.file_identity() && save_modulations(directory, path, key, result) ? "miss" : "write_failed";
    return result;
  }
}

Tensor qkv_attention(GraphSession& session, WeightStore& store, const std::string& prefix, const Tensor& x, const Tensor* cosine, const Tensor* sine) {
  Tensor fused = session.linear(store, prefix + ".qkv_proj", x);
  Tensor result = session.fused_qkv_attention(fused, store.tensor(prefix + ".q_norm.weight"),
                                               store.tensor(prefix + ".k_norm.weight"), cosine, sine);
  return session.linear(store, prefix + ".out_proj", result);
}

Tensor mlp(GraphSession& session, WeightStore& store, const std::string& prefix, const Tensor& x) {
  Tensor hidden = session.linear(store, prefix + ".fc1", x);
  return session.linear(store, prefix + ".fc2", session.gated_activation(hidden));
}

Tensor prepare_text(GraphSession& session, WeightStore& store, const Tensor& embedding) {
  Tensor x = session.linear(store, "condition_proj", embedding);
  for (uint32_t layer = 0; layer < 2; ++layer) {
    @autoreleasepool {
      const std::string prefix = "token_refiner.blocks." + std::to_string(layer) + ".";
      Tensor h = session.rms_norm(x, store.tensor(prefix + "norm1.weight"), 1e-5);
      x = session.add(x, qkv_attention(session, store, prefix + "attn", h, nullptr, nullptr));
      h = session.rms_norm(x, store.tensor(prefix + "norm2.weight"), 1e-5);
      x = session.add(x, mlp(session, store, prefix + "mlp", h));
    }
  }
  return session.rms_norm(x, store.tensor("token_refiner.final_norm.weight"), 1e-5);
}

std::pair<Tensor, Tensor> rope_angles(WeightStore& store, int64_t text_count,
                                       const std::vector<ReferenceConditioning::Grid>& grids) {
  int64_t count = text_count;
  for (const auto& [rows, columns] : grids) count += rows * columns;
  Tensor frequencies = store.tensor("rope.inv_freq");
  const auto* frequency = static_cast<const float*>(frequencies.data());
  std::vector<float> cosine(count * 96);
  std::vector<float> sine(count * 96);
  int64_t row = 0;
  const auto append_position = [&](const std::array<double, 3>& position) {
    for (uint32_t axis = 0; axis < 3; ++axis) {
      for (uint32_t index = 0; index < 16; ++index) {
        const double angle = position[axis] * frequency[index];
        for (uint32_t copy = 0; copy < 2; ++copy) {
          const uint64_t output = row * 96 + copy * 48 + axis * 16 + index;
          cosine[output] = std::cos(angle);
          sine[output] = std::sin(angle);
        }
      }
    }
    ++row;
  };
  for (int64_t token = 0; token < text_count; ++token) {
    append_position({static_cast<double>(token), 0, 0});
  }
  double time = text_count;
  for (const auto& [rows, columns] : grids) {
    const double area = std::sqrt(static_cast<double>(rows * columns));
    for (uint32_t y = 0; y < rows; ++y) {
      for (uint32_t x = 0; x < columns; ++x) {
        append_position({time, (y / area + (1 - rows / area) / 2) * 32,
                         (x / area + (1 - columns / area) / 2) * 32});
      }
    }
    ++time;
  }
  const std::vector<int64_t> shape = {1, 1, count, 96};
  return {make_tensor(shape, MPSDataTypeBFloat16, cosine), make_tensor(shape, MPSDataTypeBFloat16, sine)};
}

enum ModulationPart {
  kAttentionShift = 0,
  kAttentionGate = 2,
  kMlpShift = 3,
  kMlpGate = 5,
};

Tensor block_modulation_rows(GraphSession& session, const Tensor& table, uint32_t step) {
  // The current step supplies the target and text rows. Reference latents use the
  // image row from the last table, computed once at their fixed conditioning timestep.
  Tensor rows = table;
  const uint64_t step_bytes = table.byte_size() / table.shape[0];
  rows.byte_offset += step_bytes * step;
  rows.shape = {12, kHiddenSize};
  if (table.shape[0] > kSamplingSteps) {
    Tensor reference = table;
    reference.byte_offset += step_bytes * kSamplingSteps;
    reference.shape = {6, kHiddenSize};
    rows = session.concat({rows, reference}, 0);
  }
  return rows;
}

}  // namespace

Tensor denoise_image(const std::string& model, const std::string& embedding_path, const std::string& noise_path, const std::string& pdd_path, const std::string& output, uint32_t width, uint32_t height, const StageOptions& options,
                     const std::string& references_path) {
  if (width % 32 || height % 32) throw std::runtime_error("H3 dimensions must be multiples of 32");
  require_streaming_weights(options, "dit", "W4 DiT pack 37.47 GB exceeds physical memory");
  if (options.prefetch_weights && options.cache_bytes != 0) {
    throw std::runtime_error("DiT prefetch requires --cache-bytes 0");
  }
  const auto began = std::chrono::steady_clock::now();
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  GraphSession session(device);
  if (std::getenv("ONYX_PROFILE")) session.enable_profiling(true);
  WeightStore store(model, false, options.cache_bytes);
  WeightStore pdd(pdd_path);
  const auto sigmas = read_sigmas(model);
  std::vector<float> modulation_sigmas(sigmas.begin(), sigmas.begin() + kSamplingSteps);
  if (!references_path.empty()) modulation_sigmas.push_back(kReferenceSigma);
  const char* modulation_cache_status = nullptr;
  const auto modulation_began = std::chrono::steady_clock::now();
  // Without an on-disk modulation cache, the first sampling step computes each table right
  // before its block uses it. The 520 MB projection weights then stream alongside the block
  // weights instead of in a serial pass ahead of sampling.
  const bool lazy_modulation = options.modulation_cache_dir.empty() && options.prefetch_weights;
  std::vector<Tensor> modulations;
  Tensor temb{};
  if (lazy_modulation) {
    modulation_cache_status = "disabled";
    temb = time_embedding(session, store, modulation_sigmas, true);
    modulations.assign(kDiTBlocks + 1, Tensor{});
    queue_modulation_table(store, 0);
    queue_modulation_table(store, 1);
  } else {
    modulations = get_modulations(session, store, modulation_sigmas, device, options.modulation_cache_dir,
                                  options.prefetch_weights, modulation_cache_status);
  }
  const double modulation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - modulation_began).count();
  const Tensor& final_modulation = modulations.back();
  Tensor text = prepare_text(session, store, load_tensor(embedding_path, "embedding"));
  const uint32_t rows = height / 32;
  const uint32_t columns = width / 32;
  const int64_t text_count = text.shape[0];
  const int64_t image_count = rows * columns;
  auto references = load_references(references_path, text_count);
  const int64_t reference_patch_count = references.grids.empty() ? 0 : references.patches.shape[0];
  const int64_t target_start = text_count + reference_patch_count;
  const int64_t sequence_count = target_start + image_count;
  const size_t reference_images = references.grids.size();
  const char* indexed_modulation_env = std::getenv("ONYX_DIT_INDEXED_MODULATION");
  const bool indexed_modulation = reference_images && indexed_modulation_env &&
      std::string_view(indexed_modulation_env) == "1";
  const char* indexed_norm_env = std::getenv("ONYX_DIT_INDEXED_NORM");
  const bool text_indexed_norm = !reference_images && indexed_norm_env &&
      std::string_view(indexed_norm_env) == "1";
  const bool indexed_norm = indexed_modulation || text_indexed_norm;
  references.grids.emplace_back(rows, columns);
  auto [cosine, sine] = rope_angles(store, text_count, references.grids);
  Tensor token_indices = make_tensor({sequence_count}, MPSDataTypeInt32, std::vector<float>(sequence_count, 0));
  auto* tags = static_cast<int32_t*>(token_indices.data());
  if (reference_patch_count > 0) {
    std::memcpy(tags, references.text_tags.data(), text_count * sizeof(int32_t));
    std::fill(tags + text_count, tags + target_start, 2);
    Tensor image = session.linear(store, "video_patch_proj", references.patches);
    text = session.concat({text, session.cast(image, MPSDataTypeBFloat16)}, 0);
  } else {
    std::fill(tags, tags + text_count, 1);
  }
  const Tensor one = scalar(1.0f, MPSDataTypeBFloat16);
  Tensor patches = load_tensor(noise_path, "latent");
  patches = session.reshape(patches, {24, static_cast<int64_t>(rows), 2, static_cast<int64_t>(columns), 2});
  patches = session.transpose(patches, {1, 3, 0, 2, 4});
  patches = session.reshape(patches, {static_cast<int64_t>(image_count), 96});
  // Keep the reader one block ahead. The pipeline stays full across sampling-step boundaries,
  // where the next step's patch projection and block 0 follow the last block.
  const auto queue_block = [&](uint32_t layer) {
    if (!options.prefetch_weights) return;
    if (layer == 0) store.prefetch("video_patch_proj", kLinearSuffixes);
    store.prefetch("blocks." + std::to_string(layer) + ".", kBlockTensors);
  };
  queue_block(0);
  for (uint32_t step = 0; step < 6; ++step) {
    @autoreleasepool {
    const auto step_began = std::chrono::steady_clock::now();
    Tensor image = session.linear(store, "video_patch_proj", patches);
    image = session.cast(image, MPSDataTypeBFloat16);
    Tensor x = session.concat({text, image}, 0);
    // The AdaLN tables of the first sampling step arrive one block at a time.
    if (lazy_modulation && step == 0 && !modulations[0].buffer) {
      modulations[0] = modulation_table(session, store, 0, temb);
    }
    Tensor table{};
    std::vector<Tensor> parts;
    const auto prepare_modulation = [&](uint32_t layer) {
      table = block_modulation_rows(session, modulations[layer], step);
      if (text_indexed_norm) {
        parts = session.modulation_gates(table, token_indices);
      } else if (!indexed_modulation) {
        parts = session.modulation_parts(table, token_indices);
      }
    };
    const auto modulate = [&](const Tensor& input, const Tensor& norm, ModulationPart shift) {
      return indexed_norm
          ? session.indexed_modulated_rms_norm(input, norm, table, token_indices, shift, 1e-5)
          : session.modulated_rms_norm(input, norm, parts[shift + 1], parts[shift], 1e-5);
    };
    const auto residual = [&](const Tensor& input, const Tensor& value, ModulationPart gate) {
      if (indexed_modulation) {
        return session.indexed_gated_residual(input, value, table, token_indices, gate);
      }
      const size_t part = text_indexed_norm ? (gate == kAttentionGate ? 0 : 1) : size_t(gate);
      return session.gated_residual(input, value, parts[part]);
    };
    prepare_modulation(0);
    // The norm that opens each block follows the gated residual that closed the one before.
    Tensor h = modulate(x, store.tensor("blocks.0.norm1.weight"), kAttentionShift);
    for (uint32_t layer = 0; layer < kDiTBlocks; ++layer) {
      @autoreleasepool {
      if (layer + 1 < kDiTBlocks || step + 1 < 6) queue_block((layer + 1) % kDiTBlocks);
      if (lazy_modulation && step == 0) {
        // The first step fills the tables and the other five reuse them. The next block's
        // table is needed at the end of this block, where its first norm already begins.
        queue_modulation_table(store, layer + 2);
        if (layer + 1 < kDiTBlocks && !modulations[layer + 1].buffer) {
          modulations[layer + 1] = modulation_table(session, store, layer + 1, temb);
        }
      }
      const std::string prefix = "blocks." + std::to_string(layer) + ".";
      Tensor attended = qkv_attention(session, store, prefix + "attn", h, &cosine, &sine);
      x = residual(x, attended, kAttentionGate);
      h = modulate(x, store.tensor(prefix + "norm2.weight"), kMlpShift);
      Tensor hidden = mlp(session, store, prefix + "mlp", h);
      x = residual(x, hidden, kMlpGate);
      if (layer + 1 < kDiTBlocks) {
        prepare_modulation(layer + 1);
        h = modulate(x, store.tensor("blocks." + std::to_string(layer + 1) + ".norm1.weight"), kAttentionShift);
      }
      }
    }
    if (lazy_modulation && step == 0 && !modulations[kDiTBlocks].buffer) {
      modulations[kDiTBlocks] = modulation_table(session, store, kDiTBlocks, temb);
    }
    Tensor target = session.slice(x, 0, target_start, image_count);
    target = session.rms_norm(target, store.tensor("final_layer.norm.weight"), 1e-5);
    Tensor final = session.reshape(session.slice(final_modulation, 0, step, 1), {2, 5376});
    Tensor shift = session.slice(final, 0, 0, 1);
    Tensor scale = session.slice(final, 0, 1, 1);
    target = session.add(session.multiply(target, session.add(one, scale)), shift);
    target = session.cast(target, MPSDataTypeFloat32);
    Tensor head = session.reshape(session.slice(pdd.tensor("video_out.weight"), 0, step, 1), {96, 5376});
    Tensor bias = session.reshape(session.slice(pdd.tensor("video_out.bias"), 0, step, 1), {96});
    Tensor velocity = session.matmul(target, head);
    velocity = session.add(velocity, bias);
    Tensor delta = scalar(sigmas[step] - sigmas[step + 1], MPSDataTypeFloat32);
    patches = session.add(patches, session.multiply(velocity, delta));
    const double step_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - step_began).count();
    std::fprintf(stderr, "native DiT step %u/6, %.3f s, Metal allocated %.2f GiB\n", step + 1, step_seconds,
                 device.currentAllocatedSize / 1073741824.0);
    }
  }
  if (options.prefetch_weights) store.finish_prefetch();
  Tensor latent = session.reshape(patches, {static_cast<int64_t>(rows), static_cast<int64_t>(columns), 24, 2, 2});
  latent = session.transpose(latent, {2, 0, 3, 1, 4});
  latent = session.reshape(latent, {24, static_cast<int64_t>(rows * 2), static_cast<int64_t>(columns * 2)});
  save_tensor(output, "latent", latent);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  std::printf("{\"stage\":\"dit\",\"profile\":\"%s\",\"steps\":6,\"seconds\":%.6f,\"output\":\"%s\","
              "\"reference_images\":%zu,\"indexed_modulation\":%s,\"norm_mode\":\"%s\",\"gate_mode\":\"%s\","
              "\"modulation_cache\":\"%s\",\"modulation_seconds\":%.6f,\"lazy_modulation\":%s,"
              "\"prefetch_weights\":%s,\"prefetch_wait_seconds\":%.6f,"
              "\"weight_cache\":{\"budget\":%llu,\"bytes\":%llu,\"entries\":%llu,\"hits\":%llu,\"read_calls\":%llu,\"read_bytes\":%llu}}\n",
              options.profile.c_str(), seconds, output.c_str(), reference_images, indexed_modulation ? "true" : "false",
              indexed_norm ? "indexed" : "gathered", indexed_modulation ? "indexed" : "gathered",
              modulation_cache_status, modulation_seconds,
              lazy_modulation ? "true" : "false",
              options.prefetch_weights ? "true" : "false", store.prefetch_wait_seconds(),
              static_cast<unsigned long long>(store.cache_budget()),
              static_cast<unsigned long long>(store.cache_bytes()),
              static_cast<unsigned long long>(store.cache_entries()),
              static_cast<unsigned long long>(store.cache_hits()),
              static_cast<unsigned long long>(store.read_calls()),
              static_cast<unsigned long long>(store.bytes_read()));
  if (std::getenv("ONYX_PROFILE")) {
    std::printf("{\"profile\":%s,\"weight_read\":{\"bytes\":%llu,\"calls\":%llu,\"seconds\":%.6f}}\n",
                session.profile_json().c_str(), store.bytes_read(), store.read_calls(), store.read_seconds());
    std::vector<std::pair<std::string, uint64_t>> reads(store.read_bytes_by_name().begin(), store.read_bytes_by_name().end());
    std::sort(reads.begin(), reads.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (size_t index = 0; index < std::min<size_t>(reads.size(), 8); ++index) {
      std::fprintf(stderr, "read-bytes %s %llu\n", reads[index].first.c_str(), reads[index].second);
    }
  }
  return latent;
}

}  // namespace onyx
