#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>

#ifdef ONYX_MLX_BACKEND
#import <mlx/mlx.h>
#endif

#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace onyx {

// Each native stage takes --profile and --weights explicitly.
// Missing or invalid values fail during stage argument parsing.
struct StageOptions {
  std::string profile;
  std::string weights;  // "resident" | "stream"
  uint64_t cache_bytes = 0;  // DiT LRU budget in bytes. Zero selects streaming.
  std::string modulation_cache_dir;
  bool prefetch_weights = false;
};

StageOptions parse_stage_options(int argc, char** argv);

void require_streaming_weights(const StageOptions& options, const char* stage, const char* reason);

struct Tensor {
  id<MTLBuffer> buffer;
  std::vector<int64_t> shape;
  MPSDataType dtype;
  uint64_t byte_offset = 0;

  uint64_t elements() const;
  uint64_t byte_size() const;
  void* data() const;
};

class WeightStore {
 public:
  // Resident mode retains all tensors. Otherwise, a positive cache budget enables weight LRU.
  // Scale and zero-point caching is independent of the weight budget.
  explicit WeightStore(const std::string& path, bool resident = false, uint64_t cache_budget_bytes = 0);
  ~WeightStore();
  bool has(const std::string& name) const;
  std::vector<std::string> names() const;
  std::string dtype(const std::string& name) const;
  std::vector<int64_t> shape(const std::string& name) const;
  Tensor tensor(const std::string& name);
  Tensor scale(const std::string& name);
  Tensor embedding(const std::string& name, const std::vector<int32_t>& ids);
  std::string file_identity() const;
  // Queue background reads for the tensors stored as prefix + suffix, with the scales of a
  // quantized tensor read right after it. Streaming only. tensor() and scale() take the
  // results and wait for a read that is still in flight.
  void prefetch(const std::string& prefix, const std::vector<std::string>& suffixes);
  void finish_prefetch();
  double prefetch_wait_seconds() const { return prefetch_wait_ns_ / 1e9; }
  uint64_t bytes_read() const { return bytes_read_; }
  uint64_t read_calls() const { return read_calls_; }
  double read_seconds() const { return read_ns_ / 1e9; }
  uint64_t cache_budget() const { return cache_budget_; }
  uint64_t cache_bytes() const { return cache_bytes_; }
  uint64_t cache_entries() const { return lru_bytes_.size(); }
  uint64_t cache_hits() const { return cache_hits_; }
  const std::map<std::string, uint64_t>& read_bytes_by_name() const { return read_bytes_by_name_; }

 private:
  int fd_;
  uint64_t data_begin_;
  NSDictionary* specs_;
  std::vector<std::string> names_;  // Sorted tensor names.
  id<MTLDevice> device_;
  bool resident_;
  uint64_t cache_budget_;
  uint64_t cache_bytes_ = 0;
  uint64_t cache_hits_ = 0;
  std::list<std::string> lru_order_;  // front = MRU
  std::map<std::string, uint64_t> lru_bytes_;
  std::map<std::string, Tensor> cache_;
  void admit(const std::string& name, const Tensor& value);
  void touch(const std::string& name);
  uint64_t bytes_read_ = 0;
  uint64_t read_calls_ = 0;
  uint64_t read_ns_ = 0;
  std::map<std::string, uint64_t> read_bytes_by_name_;
  std::mutex read_stats_mutex_;
  // Reader pipeline. The reader threads share only the queue, the in-flight count, the
  // results, and the errors. The caches above stay on the calling thread.
  struct Request {
    std::string name;
    bool scales = false;
  };
  Tensor read_scale(const std::string& name);
  Tensor take_prefetched(const std::string& key);
  void read_loop();
  std::mutex prefetch_mutex_;
  std::condition_variable work_available_;
  std::condition_variable result_ready_;
  std::deque<Request> prefetch_queue_;
  std::set<std::string> prefetch_pending_;  // Queued, in flight, or ready and not yet taken
  std::map<std::string, Tensor> prefetched_;
  std::map<std::string, std::exception_ptr> prefetch_errors_;
  size_t prefetch_active_ = 0;
  bool prefetch_stop_ = false;
  std::vector<std::thread> prefetch_readers_;
  uint64_t prefetch_wait_ns_ = 0;
  Tensor read(const std::string& name, const std::string& field, bool scales);
};

MPSDataType data_type(const std::string& name);
std::vector<int64_t> shape_of(NSArray* values);
void save_tensor(const std::string& path, const std::string& name, const Tensor& value);
Tensor load_tensor(const std::string& path, const std::string& name);
std::vector<int32_t> read_token_ids(const std::string& path);
uint16_t float_to_bf16(float value);
float bf16_to_float(uint16_t value);
uint16_t float_to_fp16_bits(float value);
float fp16_bits_to_float(uint16_t bits);

class GraphSession {
 public:
  explicit GraphSession(id<MTLDevice> device);
  Tensor linear(WeightStore& store, const std::string& prefix, const Tensor& x);
  Tensor matmul(const Tensor& x, const Tensor& weight);
  Tensor rms_norm(const Tensor& x, const Tensor& weight, double epsilon);
  Tensor modulated_rms_norm(const Tensor& x, const Tensor& weight, const Tensor& scale, const Tensor& shift, double epsilon);
  Tensor indexed_modulated_rms_norm(const Tensor& x, const Tensor& weight, const Tensor& table,
                                    const Tensor& indices, uint32_t shift_part, double epsilon);
  Tensor indexed_gated_residual(const Tensor& x, const Tensor& value, const Tensor& table,
                                const Tensor& indices, uint32_t gate_part);
  Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, double epsilon);
  Tensor rope(const Tensor& x, const Tensor& cosine, const Tensor& sine);
  Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal, double scale);
  Tensor fused_qkv_attention(const Tensor& qkv, const Tensor& q_weight, const Tensor& k_weight, const Tensor* cosine, const Tensor* sine);
  Tensor tile_attention(const Tensor& qkv, const Tensor& cosine, const Tensor& sine);
  Tensor reshape(const Tensor& x, const std::vector<int64_t>& shape);
  Tensor transpose(const Tensor& x, const std::vector<uint32_t>& permutation);
  Tensor slice(const Tensor& x, uint32_t dimension, int64_t start, int64_t length);
  Tensor concat(const std::vector<Tensor>& values, uint32_t dimension);
  Tensor add(const Tensor& x, const Tensor& y);
  Tensor multiply(const Tensor& x, const Tensor& y);
  Tensor silu(const Tensor& x);
  Tensor gated_activation(const Tensor& x);
  Tensor gated_residual(const Tensor& x, const Tensor& value, const Tensor& gate);
  // Gather one [tokens, width] tensor per AdaLN part from `rows`, which holds the parts of
  // the image modality followed by those of the text modality. Token index 0 selects the
  // image rows and 1 the text rows.
  std::vector<Tensor> modulation_parts(const Tensor& rows, const Tensor& indices);
  std::vector<Tensor> modulation_gates(const Tensor& rows, const Tensor& indices);
  Tensor cast(const Tensor& x, MPSDataType dtype);
  void enable_profiling(bool enabled) { profile_enabled_ = enabled; }
  std::string profile_json() const;

 private:
  id<MTLCommandQueue> queue_;
  NSMutableDictionary* graphs_;
  std::map<std::string, Tensor> zero_cache_;
  bool profile_enabled_ = false;
  std::map<std::string, std::pair<uint64_t, uint64_t>> profile_;
  // Cached scales, biases, and pipeline state for the optional Steel and SIMD paths.
  std::map<std::string, Tensor> steel_scales_;
  std::map<std::string, Tensor> steel_biases_;
  id<MTLComputePipelineState> steel_pipeline_ = nil;
  id<MTLComputePipelineState> simd_pipeline_ = nil;
  id<MTLBuffer> simd_zero_bias_;
  id<MTLBuffer> steel_meta_ = nil;  // Placeholder metadata for the unbatched Steel kernel.
  Tensor steel_scales_bf16(const std::string& name, const Tensor& scales);
  Tensor steel_biases_bf16(const std::string& name, const Tensor& scales_bf16);
  Tensor linear_steel(WeightStore& store, const std::string& weight_name, const Tensor& x,
                      uint32_t n, uint32_t k, bool has_bias, const Tensor& bias);
  Tensor linear_q4_simd(WeightStore& store, const std::string& weight_name, const Tensor& x,
                        uint32_t n, uint32_t k, bool has_bias, const Tensor& bias);
  // Q4 linear layers on the M5 neural accelerators, in q4_nax.mm.
  bool nax_available();
  Tensor linear_q4_nax(const std::string& weight_name, const Tensor& weight, const Tensor& scales,
                       const Tensor& x, int64_t n, int64_t k);
  std::map<std::string, id<MTLComputePipelineState>> nax_pipelines_;
  // BF16 scales and affine biases per weight, which is how the NAX kernel takes them.
  std::map<std::string, std::pair<Tensor, Tensor>> nax_affine_;
#ifdef ONYX_MLX_BACKEND
  struct MlxQ4Weights {
    mlx::core::array w;
    mlx::core::array s;
    mlx::core::array b;
    int64_t k = 0;
    int64_t n = 0;
    int64_t group_size = 0;
  };
  std::unordered_map<std::string, MlxQ4Weights> mlx_weight_cache_;
#endif
  std::vector<Tensor> run_outputs(const std::string& key,
                                  const std::vector<Tensor>& inputs,
                                  const std::vector<std::vector<int64_t>>& output_shapes,
                                  MPSDataType output_dtype,
                                  NSArray<MPSGraphTensor*>* (^build)(MPSGraph*, NSArray<MPSGraphTensor*>*));
  Tensor run(const std::string& key,
             const std::vector<Tensor>& inputs,
             const std::vector<int64_t>& output_shape,
             MPSDataType output_dtype,
             MPSGraphTensor* (^build)(MPSGraph*, NSArray<MPSGraphTensor*>*));
};

Tensor encode_text(const std::string& model, const std::string& tokens, const std::string& output, const StageOptions& options,
    const std::string& vision_path = {});
Tensor denoise_image(const std::string& model, const std::string& embedding, const std::string& noise, const std::string& pdd, const std::string& output, uint32_t width, uint32_t height, const StageOptions& options,
    const std::string& references_path = {});
Tensor decode_image(const std::string& model, const std::string& latent, const std::string& config, const std::string& output, const StageOptions& options);

}  // namespace onyx
