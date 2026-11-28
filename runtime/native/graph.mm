#import "native.hpp"

#import <MPSNDArray/MPSNDArray.h>
#import <MetalPerformanceShadersGraph/MPSGraphActivationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphArithmeticOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphGatherOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMemoryOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphNormalizationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphQuantizationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>

#ifdef ONYX_MLX_BACKEND
#import <mlx/mlx.h>
#import <mlx/backend/metal/metal.h>
namespace {
namespace mx = mlx::core;
}
#endif

#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace onyx {
namespace {

NSArray* ns_shape(const std::vector<int64_t>& shape) {
  NSMutableArray* result = [NSMutableArray arrayWithCapacity:shape.size()];
  for (int64_t value : shape) [result addObject:@(value)];
  return result;
}

std::string shape_key(const Tensor& value) {
  std::ostringstream out;
  out << ':' << value.dtype << ':';
  for (int64_t dimension : value.shape) out << dimension << ',';
  return out.str();
}

MPSGraphTensor* cast_to(MPSGraph* graph, MPSGraphTensor* value, MPSDataType dtype) {
  return value.dataType == dtype ? value : [graph castTensor:value toType:dtype name:nil];
}

MPSGraphTensor* rms_unit(MPSGraph* graph, MPSGraphTensor* value, double epsilon) {
  const NSInteger axis = static_cast<NSInteger>(value.shape.count) - 1;
  MPSGraphTensor* square = [graph squareWithTensor:value name:nil];
  MPSGraphTensor* mean = [graph meanOfTensor:square axes:@[@(axis)] name:nil];
  MPSGraphTensor* eps = [graph constantWithScalar:epsilon shape:@[@1] dataType:value.dataType];
  MPSGraphTensor* sum = [graph additionWithPrimaryTensor:mean secondaryTensor:eps name:nil];
  MPSGraphTensor* inverse = [graph reciprocalSquareRootWithTensor:sum name:nil];
  return [graph multiplicationWithPrimaryTensor:value secondaryTensor:inverse name:nil];
}

// The compact modulation table has six consecutive rows for each modality.
MPSGraphTensor* modulation_part(MPSGraph* graph, MPSGraphTensor* table,
                                MPSGraphTensor* base, uint32_t part) {
  MPSGraphTensor* offset = [graph constantWithScalar:part shape:@[@1] dataType:MPSDataTypeInt32];
  MPSGraphTensor* index = [graph additionWithPrimaryTensor:base secondaryTensor:offset name:nil];
  return [graph gatherWithUpdatesTensor:table indicesTensor:index axis:0 batchDimensions:0 name:nil];
}

MPSGraphTensor* modulation_base(MPSGraph* graph, MPSGraphTensor* indices) {
  MPSGraphTensor* stride = [graph constantWithScalar:6 shape:@[@1] dataType:MPSDataTypeInt32];
  return [graph multiplicationWithPrimaryTensor:indices secondaryTensor:stride name:nil];
}

MPSGraphTensor* modulated_norm(MPSGraph* graph, MPSGraphTensor* x, MPSGraphTensor* weight,
                               MPSGraphTensor* scale, MPSGraphTensor* shift, double epsilon) {
  MPSGraphTensor* w = cast_to(graph, weight, x.dataType);
  MPSGraphTensor* normalized = [graph multiplicationWithPrimaryTensor:rms_unit(graph, x, epsilon)
                                                     secondaryTensor:w name:nil];
  MPSGraphTensor* one = [graph constantWithScalar:1.0 shape:@[@1] dataType:scale.dataType];
  MPSGraphTensor* modulation = [graph additionWithPrimaryTensor:one secondaryTensor:scale name:nil];
  MPSGraphTensor* y = [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:modulation name:nil];
  return [graph additionWithPrimaryTensor:y secondaryTensor:shift name:nil];
}

MPSGraphTensor* residual_with_gate(MPSGraph* graph, MPSGraphTensor* x,
                                   MPSGraphTensor* value, MPSGraphTensor* gate) {
  MPSGraphTensor* scale = cast_to(graph, gate, value.dataType);
  MPSGraphTensor* scaled = [graph multiplicationWithPrimaryTensor:value secondaryTensor:scale name:nil];
  return [graph additionWithPrimaryTensor:x secondaryTensor:scaled name:nil];
}

MPSGraphTensor* apply_rope(MPSGraph* graph, MPSGraphTensor* value, MPSGraphTensor* cosine, MPSGraphTensor* sine) {
  const int64_t dimension = value.shape.lastObject.longLongValue;
  const int64_t rotary = cosine.shape.lastObject.longLongValue;
  MPSGraphTensor* first = [graph sliceTensor:value dimension:3 start:0 length:rotary / 2 name:nil];
  MPSGraphTensor* second = [graph sliceTensor:value dimension:3 start:rotary / 2 length:rotary / 2 name:nil];
  MPSGraphTensor* negative = [graph negativeWithTensor:second name:nil];
  MPSGraphTensor* rotated = [graph concatTensors:@[negative, first] dimension:3 interleave:NO name:nil];
  MPSGraphTensor* original = [graph concatTensors:@[first, second] dimension:3 interleave:NO name:nil];
  MPSGraphTensor* a = [graph multiplicationWithPrimaryTensor:original secondaryTensor:cosine name:nil];
  MPSGraphTensor* b = [graph multiplicationWithPrimaryTensor:rotated secondaryTensor:sine name:nil];
  MPSGraphTensor* result = [graph additionWithPrimaryTensor:a secondaryTensor:b name:nil];
  if (rotary < dimension) {
    MPSGraphTensor* pass = [graph sliceTensor:value dimension:3 start:rotary length:dimension - rotary name:nil];
    result = [graph concatTensors:@[result, pass] dimension:3 interleave:NO name:nil];
  }
  return result;
}

MPSGraphTensor* scaled_attention(MPSGraph* graph, MPSGraphTensor* query,
                                 MPSGraphTensor* key, MPSGraphTensor* value,
                                 bool causal, float scale) {
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 270000
  if (@available(macOS 27.0, *)) {
    MPSGraphSDPADescriptor* descriptor = [MPSGraphSDPADescriptor descriptorWithScale:scale];
    descriptor.isCausal = causal;
    return [graph scaledDotProductAttentionWithQueryTensor:query keyTensor:key valueTensor:value
                                                descriptor:descriptor name:nil];
  }
#endif
  MPSGraphTensor* mask = nil;
  if (causal) {
    NSNumber* query_length = query.shape[query.shape.count - 2];
    NSNumber* key_length = key.shape[key.shape.count - 2];
    MPSGraphTensor* row = [graph coordinateAlongAxis:0 withShape:@[query_length, @1] name:nil];
    MPSGraphTensor* column = [graph coordinateAlongAxis:1 withShape:@[@1, key_length] name:nil];
    MPSGraphTensor* allowed = [graph lessThanOrEqualToWithPrimaryTensor:column secondaryTensor:row name:nil];
    MPSGraphTensor* zero = [graph constantWithScalar:0 dataType:query.dataType];
    MPSGraphTensor* masked = [graph constantWithScalar:-std::numeric_limits<float>::infinity() dataType:query.dataType];
    mask = [graph selectWithPredicateTensor:allowed truePredicateTensor:zero falsePredicateTensor:masked name:nil];
  }
  return [graph scaledDotProductAttentionWithQueryTensor:query keyTensor:key valueTensor:value
                                              maskTensor:mask scale:scale name:nil];
}

std::vector<int64_t> broadcast_shape(const Tensor& x, const Tensor& y) {
  const size_t rank = std::max(x.shape.size(), y.shape.size());
  std::vector<int64_t> result(rank, 1);
  for (size_t offset = 0; offset < rank; ++offset) {
    const int64_t a = offset < rank - x.shape.size() ? 1 : x.shape[offset - (rank - x.shape.size())];
    const int64_t b = offset < rank - y.shape.size() ? 1 : y.shape[offset - (rank - y.shape.size())];
    if (a != b && a != 1 && b != 1) throw std::runtime_error("Incompatible broadcast shapes");
    result[offset] = std::max(a, b);
  }
  return result;
}

}  // namespace

GraphSession::GraphSession(id<MTLDevice> device) : queue_([device newCommandQueue]), graphs_([NSMutableDictionary new]) {
  if (!queue_) throw std::runtime_error("Could not create Metal command queue");
}

Tensor GraphSession::run(const std::string& key,
                         const std::vector<Tensor>& inputs,
                         const std::vector<int64_t>& output_shape,
                         MPSDataType output_dtype,
                         MPSGraphTensor* (^build)(MPSGraph*, NSArray<MPSGraphTensor*>*)) {
  const std::vector<Tensor> results = run_outputs(
      key, inputs, {output_shape}, output_dtype,
      ^NSArray<MPSGraphTensor*>*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) { return @[build(graph, input)]; });
  return results.front();
}

std::vector<Tensor> GraphSession::run_outputs(const std::string& key,
                                              const std::vector<Tensor>& inputs,
                                              const std::vector<std::vector<int64_t>>& output_shapes,
                                              MPSDataType output_dtype,
                                              NSArray<MPSGraphTensor*>* (^build)(MPSGraph*, NSArray<MPSGraphTensor*>*)) {
  const auto profile_start = std::chrono::steady_clock::now();
  std::string full_key = key;
  for (const Tensor& input : inputs) full_key += shape_key(input);
  for (const std::vector<int64_t>& shape : output_shapes) full_key += shape_key(Tensor{nil, shape, output_dtype});
  @autoreleasepool {
    NSDictionary* entry = graphs_[[NSString stringWithUTF8String:full_key.c_str()]];
    MPSGraph* graph;
    NSArray<MPSGraphTensor*>* targets;
    NSArray<MPSGraphTensor*>* placeholders;
    if (entry) {
      graph = entry[@"graph"];
      targets = entry[@"targets"];
      placeholders = entry[@"inputs"];
    } else {
      graph = [MPSGraph new];
      NSMutableArray<MPSGraphTensor*>* created = [NSMutableArray new];
      for (size_t index = 0; index < inputs.size(); ++index) {
        [created addObject:[graph placeholderWithShape:ns_shape(inputs[index].shape)
                                              dataType:inputs[index].dtype
                                                  name:[NSString stringWithFormat:@"x%zu", index]]];
      }
      targets = build(graph, created);
      if (!targets) throw std::runtime_error("Could not build graph: " + key);
      graphs_[[NSString stringWithUTF8String:full_key.c_str()]] = @{@"graph": graph, @"targets": targets, @"inputs": created};
      placeholders = created;
    }
    NSMutableDictionary* feeds = [NSMutableDictionary new];
    for (size_t index = 0; index < inputs.size(); ++index) {
      const Tensor& input = inputs[index];
      MPSGraphTensorData* data;
      if (input.byte_offset) {
        auto descriptor = [MPSNDArrayDescriptor descriptorWithDataType:input.dtype shape:ns_shape(input.shape)];
        auto view = [[MPSNDArray alloc] initWithBuffer:input.buffer offset:input.byte_offset descriptor:descriptor];
        data = [[MPSGraphTensorData alloc] initWithMPSNDArray:view];
      } else {
        data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input.buffer
                                                      shape:ns_shape(input.shape)
                                                   dataType:input.dtype];
      }
      feeds[placeholders[index]] = data;
    }
    std::vector<Tensor> results;
    NSMutableDictionary* outputs = [NSMutableDictionary new];
    for (size_t index = 0; index < output_shapes.size(); ++index) {
      Tensor result{nil, output_shapes[index], output_dtype};
      result.buffer = [queue_.device newBufferWithLength:result.byte_size() options:MTLResourceStorageModeShared];
      if (!result.buffer) throw std::runtime_error("Metal output allocation failed");
      outputs[targets[index]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:result.buffer shape:ns_shape(result.shape)
                                                                     dataType:output_dtype];
      results.push_back(result);
    }
    [graph runWithMTLCommandQueue:queue_ feeds:feeds targetOperations:nil resultsDictionary:outputs];
    if (profile_enabled_) {
      auto& entry = profile_[key];
      entry.first += 1;
      entry.second += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - profile_start).count();
    }
    return results;
  }
}

std::vector<Tensor> GraphSession::modulation_parts(const Tensor& rows, const Tensor& indices) {
  // Each index selects one row of six modulation parts: target image, text, or reference image.
  // A reference uses the image modality at its fixed conditioning timestep.
  const int64_t tokens = indices.shape[0];
  const int64_t width = rows.shape[1];
  constexpr int64_t parts = 6;
  std::vector<std::vector<int64_t>> shapes(parts, std::vector<int64_t>{tokens, width});
  return run_outputs("modulation-parts", {rows, indices}, shapes, rows.dtype,
                     ^NSArray<MPSGraphTensor*>*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    NSMutableArray<MPSGraphTensor*>* gathered = [NSMutableArray new];
    MPSGraphTensor* base = modulation_base(graph, input[1]);
    for (uint32_t part = 0; part < parts; ++part) {
      [gathered addObject:modulation_part(graph, input[0], base, part)];
    }
    return gathered;
  });
}

std::vector<Tensor> GraphSession::modulation_gates(const Tensor& rows, const Tensor& indices) {
  const std::vector<int64_t> shape{indices.shape[0], rows.shape[1]};
  return run_outputs("modulation-gates", {rows, indices}, {shape, shape}, rows.dtype,
                     ^NSArray<MPSGraphTensor*>*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* base = modulation_base(graph, input[1]);
    return @[modulation_part(graph, input[0], base, 2), modulation_part(graph, input[0], base, 5)];
  });
}


// MLX Steel affine QMM, vendored under ops/metal/mlx_root (MIT, Apple Inc.).

Tensor GraphSession::steel_scales_bf16(const std::string& name, const Tensor& scales) {
  auto found = steel_scales_.find(name);
  if (found != steel_scales_.end()) return found->second;
  const uint64_t count = scales.elements();
  std::vector<uint16_t> bf16(count);
  const float* src = static_cast<const float*>(scales.buffer.contents);
  for (uint64_t i = 0; i < count; ++i) bf16[i] = float_to_bf16(src[i]);
  id<MTLBuffer> buffer = [queue_.device newBufferWithBytes:bf16.data() length:count * sizeof(uint16_t) options:MTLResourceStorageModeShared];
  Tensor result{buffer, scales.shape, MPSDataTypeBFloat16};
  steel_scales_[name] = result;
  return result;
}

Tensor GraphSession::steel_biases_bf16(const std::string& name, const Tensor& scales_bf16) {
  auto found = steel_biases_.find(name);
  if (found != steel_biases_.end()) return found->second;
  // Onyx uses (nibble - 8) * scale. MLX uses scale * nibble + bias.
  // Set each affine bias to -8 times its scale.
  const uint64_t count = scales_bf16.elements();
  std::vector<uint16_t> bf16(count);
  const uint16_t* src = static_cast<const uint16_t*>(scales_bf16.buffer.contents);
  for (uint64_t i = 0; i < count; ++i) bf16[i] = float_to_bf16(-8.0f * bf16_to_float(src[i]));
  id<MTLBuffer> buffer = [queue_.device newBufferWithBytes:bf16.data() length:count * sizeof(uint16_t) options:MTLResourceStorageModeShared];
  Tensor result{buffer, scales_bf16.shape, MPSDataTypeBFloat16};
  steel_biases_[name] = result;
  return result;
}

Tensor GraphSession::linear_steel(WeightStore& store, const std::string& weight_name, const Tensor& x,
                                  uint32_t n, uint32_t k, bool has_bias, const Tensor& bias) {
  // Use the FP16 Steel instance. The Metal 4 BF16 instance failed numerical probes.
  // Convert BF16 activations and FP32 scales and affine biases to FP16.
  // Convert the FP16 kernel output back to BF16 before returning it.
  Tensor weight = store.tensor(weight_name);
  const uint32_t rows = uint32_t(x.elements() / k);
  const uint64_t x_count = x.elements();
  const uint64_t scale_count = store.scale(weight_name).elements();

  // Convert activations to FP16.
  std::vector<uint16_t> x_fp16(x_count);
  const uint16_t* x_bf = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(x.buffer.contents) + x.byte_offset);
  for (uint64_t i = 0; i < x_count; ++i) x_fp16[i] = float_to_fp16_bits(bf16_to_float(x_bf[i]));
  // float → fp16（scales / biases）
  const float* s_f = static_cast<const float*>(store.scale(weight_name).buffer.contents);
  std::vector<uint16_t> s_fp16(scale_count), b_fp16(scale_count);
  for (uint64_t i = 0; i < scale_count; ++i) {
    s_fp16[i] = float_to_fp16_bits(s_f[i]);
    b_fp16[i] = float_to_fp16_bits(-8.0f * s_f[i]);
  }

  if (!steel_pipeline_) {
    NSError* error = nil;
    const char* path = std::getenv("ONYX_METALLIB");
    if (!path) path = ONYX_Q4_METALLIB;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    id<MTLLibrary> library = [queue_.device newLibraryWithURL:url error:&error];
    if (!library) throw std::runtime_error(std::string("Could not load metallib for steel linear: ") + error.localizedDescription.UTF8String);
    id<MTLFunction> function = [library newFunctionWithName:@"affine_qmm_t_float16_t_gs_64_b_4_hgs_0_alN_0_batch_0_bm_32_bn_32"];
    if (!function) throw std::runtime_error("steel affine_qmm_t fp16 kernel not found in metallib");
    steel_pipeline_ = [queue_.device newComputePipelineStateWithFunction:function error:&error];
    if (!steel_pipeline_) throw std::runtime_error(std::string("steel pipeline failed: ") + error.localizedDescription.UTF8String);
    const char zeros[128] = {0};
    steel_meta_ = [queue_.device newBufferWithBytes:zeros length:128 options:MTLResourceStorageModeShared];
  }

  id<MTLBuffer> x_fp16_buffer = [queue_.device newBufferWithBytes:x_fp16.data() length:x_count * 2 options:MTLResourceStorageModeShared];
  id<MTLBuffer> s_fp16_buffer = [queue_.device newBufferWithBytes:s_fp16.data() length:scale_count * 2 options:MTLResourceStorageModeShared];
  id<MTLBuffer> b_fp16_buffer = [queue_.device newBufferWithBytes:b_fp16.data() length:scale_count * 2 options:MTLResourceStorageModeShared];
  id<MTLBuffer> y_fp16_buffer = [queue_.device newBufferWithLength:size_t(rows) * n * 2 options:MTLResourceStorageModeShared];
  if (!x_fp16_buffer || !s_fp16_buffer || !b_fp16_buffer || !y_fp16_buffer) throw std::runtime_error("steel linear allocation failed");

  id<MTLCommandBuffer> command = [queue_ commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:steel_pipeline_];
  // MLX bindings: 0=w, 1=scales, 2=biases, 3=x, 4=y, 5=K, 6=N, 7=M, 8–15=batch metadata.
  [encoder setBuffer:weight.buffer offset:0 atIndex:0];
  [encoder setBuffer:s_fp16_buffer offset:0 atIndex:1];
  [encoder setBuffer:b_fp16_buffer offset:0 atIndex:2];
  [encoder setBuffer:x_fp16_buffer offset:0 atIndex:3];
  [encoder setBuffer:y_fp16_buffer offset:0 atIndex:4];
  const int k_int = int(k), n_int = int(n), m_int = int(rows);
  [encoder setBytes:&k_int length:4 atIndex:5];
  [encoder setBytes:&n_int length:4 atIndex:6];
  [encoder setBytes:&m_int length:4 atIndex:7];
  [encoder setBuffer:steel_meta_ offset:0 atIndex:8];
  [encoder setBuffer:steel_meta_ offset:4 atIndex:9];
  [encoder setBuffer:steel_meta_ offset:20 atIndex:10];
  [encoder setBuffer:steel_meta_ offset:0 atIndex:11];
  [encoder setBuffer:steel_meta_ offset:4 atIndex:12];
  [encoder setBuffer:steel_meta_ offset:20 atIndex:13];
  [encoder setBuffer:steel_meta_ offset:20 atIndex:14];
  [encoder setBuffer:steel_meta_ offset:20 atIndex:15];
  const MTLSize groups = MTLSizeMake((n + 31) / 32, (rows + 31) / 32, 1);
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    throw std::runtime_error(std::string("steel linear failed: ") + command.error.localizedDescription.UTF8String);
  }

  // y fp16 → bf16
  const uint16_t* y_src = static_cast<const uint16_t*>(y_fp16_buffer.contents);
  std::vector<uint16_t> y_bf(size_t(rows) * n);
  for (uint64_t i = 0; i < y_bf.size(); ++i) y_bf[i] = float_to_bf16(fp16_bits_to_float(y_src[i]));
  id<MTLBuffer> y_buffer = [queue_.device newBufferWithBytes:y_bf.data() length:y_bf.size() * 2 options:MTLResourceStorageModeShared];
  Tensor y{y_buffer, std::vector<int64_t>{rows, n}, MPSDataTypeBFloat16};

  if (has_bias) {
    return run("linear:steel:bias", {y, bias}, std::vector<int64_t>{rows, n}, x.dtype,
               ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
      MPSGraphTensor* b = cast_to(graph, input[1], input[0].dataType);
      return [graph additionWithPrimaryTensor:input[0] secondaryTensor:b name:nil];
    });
  }
  return y;
}

Tensor GraphSession::linear_q4_simd(WeightStore& store, const std::string& weight_name, const Tensor& x,
                                     uint32_t n, uint32_t k, bool has_bias, const Tensor& bias) {
  // The BF16 kernel consumes native activations and produces BF16 output directly.
  // BF16 preserves the exponent range needed by the residual stream.
  // One encoder also applies bias through buffer 8.
  Tensor weight = store.tensor(weight_name);
  Tensor scales_t = store.scale(weight_name);
  const uint32_t rows = uint32_t(x.elements() / k);

  if (!simd_pipeline_) {
    NSError* error = nil;
    const char* path = std::getenv("ONYX_METALLIB");
    if (!path) path = ONYX_Q4_METALLIB;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    id<MTLLibrary> library = [queue_.device newLibraryWithURL:url error:&error];
    if (!library) throw std::runtime_error(std::string("simd linear: metallib load failed: ") + error.localizedDescription.UTF8String);
    id<MTLFunction> gemm_fn = [library newFunctionWithName:@"onyx_q4_gemm_simd_bn32_bk32_bf16"];  // Use BF16 matrix multiply.
    if (!gemm_fn) throw std::runtime_error("simd bf16 kernel not found in metallib");
    simd_pipeline_ = [queue_.device newComputePipelineStateWithFunction:gemm_fn error:&error];
    if (!simd_pipeline_) throw std::runtime_error(std::string("simd pipeline failed: ") + error.localizedDescription.UTF8String);
  }

  const auto simd_t0 = std::chrono::steady_clock::now();
  id<MTLBuffer> y_bf_buf = [queue_.device newBufferWithLength:size_t(rows) * n * 2 options:MTLResourceStorageModeShared];
  if (!y_bf_buf) throw std::runtime_error("simd linear output allocation failed");

  id<MTLCommandBuffer> command = [queue_ commandBuffer];
  // The BF16 kernel needs no activation or result conversion pass.
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:simd_pipeline_];
  [encoder setBuffer:x.buffer offset:x.byte_offset atIndex:0];
  [encoder setBuffer:weight.buffer offset:0 atIndex:1];
  [encoder setBuffer:scales_t.buffer offset:0 atIndex:2];
  [encoder setBuffer:y_bf_buf offset:0 atIndex:3];
  // SIMD bindings: 4=m, 5=n, 6=k, 7=group size, 8=bias.
  const int k_int = int(k), n_int = int(n), m_int = int(rows), g_int = 64;
  [encoder setBytes:&m_int length:4 atIndex:4];
  [encoder setBytes:&n_int length:4 atIndex:5];
  [encoder setBytes:&k_int length:4 atIndex:6];
  [encoder setBytes:&g_int length:4 atIndex:7];
  id<MTLBuffer> bias_buf;
  if (has_bias) {
    bias_buf = bias.buffer;
  } else {
    if (!simd_zero_bias_ || simd_zero_bias_.length < size_t(n) * 2) {
      std::vector<uint16_t> zeros(n, 0);
      simd_zero_bias_ = [queue_.device newBufferWithBytes:zeros.data() length:n * 2 options:MTLResourceStorageModeShared];
    }
    bias_buf = simd_zero_bias_;
  }
  [encoder setBuffer:bias_buf offset:has_bias ? bias.byte_offset : 0 atIndex:8];
  const MTLSize groups = MTLSizeMake((n + 31) / 32, (rows + 31) / 32, 1);
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
  [encoder endEncoding];
  // Deferred mode relies on queue ordering for dependent reads.
  // The caller waits at the existing synchronization point.
  static const bool simd_wait = std::getenv("ONYX_SIMD_NOWAIT") == nullptr;  // Default to waiting. ONYX_SIMD_NOWAIT=1 enables deferred waits.
  [command commit];
  if (simd_wait) [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    throw std::runtime_error(std::string("simd linear failed: ") + command.error.localizedDescription.UTF8String);
  }

  const auto simd_t2 = std::chrono::steady_clock::now();
  if (std::getenv("ONYX_SIMD_TRACE")) {
    static double gpu_total = 0;
    if (command.GPUEndTime > command.GPUStartTime) {
      gpu_total += (command.GPUEndTime - command.GPUStartTime) * 1000.0;
    }
    static int calls = 0;
    if (++calls % 224 == 0) {
      const double total_ms = std::chrono::duration<double, std::milli>(simd_t2 - simd_t0).count();
      std::fprintf(stderr, "simd timing: calls=%d host_total=%.0fms gpu_busy=%.0fms\n",
                   calls, total_ms, gpu_total);
    }
  }

  Tensor y{y_bf_buf, std::vector<int64_t>{rows, n}, MPSDataTypeBFloat16};
  return y;
}

Tensor GraphSession::linear(WeightStore& store, const std::string& prefix, const Tensor& x) {
  const std::string weight_name = prefix + ".weight";
  const char* linear_override = std::getenv("ONYX_Q4_LINEAR");
  const char* backend_override = std::getenv("ONYX_Q4_BACKEND");
  const std::string mode = linear_override && *linear_override ? linear_override
      : backend_override && *backend_override ? backend_override : "mpsgraph";
  Tensor weight = store.tensor(weight_name);
  const bool quantized = weight.dtype == MPSDataTypeUInt4;
  const bool has_bias = store.has(prefix + ".bias");
  auto output_shape = x.shape;
  output_shape.back() = weight.shape[0];
  if (!quantized) {
    std::vector<Tensor> inputs = {x, weight};
    if (has_bias) inputs.push_back(store.tensor(prefix + ".bias"));
    return run("linear:raw:" + std::to_string(has_bias), inputs, output_shape, x.dtype,
               ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
      MPSGraphTensor* w = cast_to(graph, input[1], input[0].dataType);
      w = [graph transposeTensor:w permutation:@[@1, @0] name:nil];
      MPSGraphTensor* y = [graph matrixMultiplicationWithPrimaryTensor:input[0] secondaryTensor:w name:nil];
      if (has_bias) {
        MPSGraphTensor* bias = cast_to(graph, input[2], input[0].dataType);
        y = [graph additionWithPrimaryTensor:y secondaryTensor:bias name:nil];
      }
      return y;
    });
  }
  const std::string type = store.dtype(weight_name);
  const uint64_t group_size = std::stoull(type.substr(3));
  Tensor scales = store.scale(weight_name);

  const int64_t k = x.shape.back();
  const int64_t n = weight.shape[0];
  const int64_t rows = x.elements() / k;
  if (mode == "nax" && x.dtype == MPSDataTypeBFloat16 && group_size == 64 && k % 64 == 0 && nax_available()) {
    Tensor result = linear_q4_nax(weight_name, weight, scales, x, n, k);
    result.shape = output_shape;
    if (!has_bias) return result;
    return run("bias_add", {result, store.tensor(prefix + ".bias")}, output_shape, x.dtype,
               ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
                 return [graph additionWithPrimaryTensor:input[0] secondaryTensor:cast_to(graph, input[1], input[0].dataType) name:nil];
               });
  }
  if ((mode == "simd" || mode == "steel") && x.dtype == MPSDataTypeBFloat16 && x.byte_offset == 0 &&
      group_size == 64 && k % 64 == 0 && n % 32 == 0) {
    // Keep the per-layer profiling overrides available for the experimental kernels.
    const char* skip = std::getenv("ONYX_SIMD_SKIP");
    const bool skip_layer = skip && *skip && weight_name.find(skip) != std::string::npos;
    const bool skip_odd = std::getenv("ONYX_SIMD_SKIP_ODD") && weight_name.starts_with("blocks.") &&
        std::atoi(weight_name.c_str() + 7) % 2 == 1;
    Tensor bias = has_bias ? store.tensor(prefix + ".bias") : Tensor{};
    const bool compatible_bias = !has_bias ||
        (bias.dtype == MPSDataTypeBFloat16 && bias.shape == std::vector<int64_t>{n});
    if (!skip_layer && !skip_odd && (mode == "steel" || compatible_bias)) {
      Tensor result = mode == "simd"
          ? linear_q4_simd(store, weight_name, x, n, k, has_bias, bias)
          : linear_steel(store, weight_name, x, n, k, has_bias, bias);
      result.shape = output_shape;
      return result;
    }
  }

  // The opt-in FP16 backends follow the measured shape dispatch. Keep BF16
  // tensors on BF16-compatible paths.
#ifdef ONYX_MLX_BACKEND
  if ((mode == "mlx" || mode == "auto") && x.dtype == MPSDataTypeFloat16 &&
      (group_size == 32 || group_size == 64 || group_size == 128) && k % group_size == 0) {
    static dispatch_once_t mlx_once;
    dispatch_once(&mlx_once, ^{
      mx::set_default_device(mx::Device(mx::Device::gpu));
      mx::metal::set_metallib_path(MLX_METALLIB_PATH);
    });
    // Weight cache: one-time repack bytes -> uint32 words + derived biases.
    auto cached = mlx_weight_cache_.find(weight_name);
    const bool mlx_debug = getenv("ONYX_MLX_DEBUG") != nullptr;
    if (mlx_debug) std::fprintf(stderr, "mlx path: %s rows=%lld k=%lld w.data=%p off=%llu x.data=%p x.off=%llu\n",
        weight_name.c_str(), (long long)rows, (long long)k, weight.data(), (unsigned long long)weight.byte_offset, x.data(), (unsigned long long)x.byte_offset);
    if (cached == mlx_weight_cache_.end()) {
      const uint8_t* packed_bytes = static_cast<const uint8_t*>(weight.data());
      const size_t word_count = size_t(n) * k / 8;
      std::vector<uint32_t> words(word_count);
      for (size_t index = 0; index < word_count; ++index) {
        words[index] = uint32_t(packed_bytes[index * 4 + 0]) | (uint32_t(packed_bytes[index * 4 + 1]) << 8) |
                       (uint32_t(packed_bytes[index * 4 + 2]) << 16) | (uint32_t(packed_bytes[index * 4 + 3]) << 24);
      }
      const float* scale_data = static_cast<const float*>(scales.data());
      const size_t scale_count = scales.elements();
      std::vector<float> bias_data(scale_count);
      for (size_t index = 0; index < scale_count; ++index) bias_data[index] = -8.0f * scale_data[index];
      mx::array w = mx::array(words.data(), {int(n), int(k / 8)}, mx::uint32);
      mx::array s = mx::array(scale_data, {int(n), int(k / int(group_size))}, mx::float32);
      mx::array b = mx::array(bias_data.data(), {int(n), int(k / int(group_size))}, mx::float32);
      if (mlx_debug) std::fprintf(stderr, "mlx: cache built for %s\n", weight_name.c_str());
      mx::eval(w, s, b);
      cached = mlx_weight_cache_.emplace(weight_name, MlxQ4Weights{w, s, b, k, n, int64_t(group_size)}).first;
    }
    // Copy activations into an MLX-owned array before dispatch.
    std::vector<float> activation_f(size_t(rows) * k);
    {
      const __fp16* source = static_cast<const __fp16*>(x.data());
      for (size_t index = 0; index < activation_f.size(); ++index) {
        activation_f[index] = float(source[index]);
      }
    }
    mx::array x_mlx = mx::array(activation_f.data(), {int(rows), int(k)}, mx::float16);
    mx::array y = mx::quantized_matmul(x_mlx, cached->second.w, cached->second.s, cached->second.b,
                                       /*transpose*/ true, /*group_size*/ int(group_size), /*bits*/ 4);
    // Force host sync through an fp32 cast (reading an unevaluated fp16
    // GPU array directly returns garbage; standalone-probe evidence).
    mx::array y_f = mx::astype(y, mx::float32);
    mx::eval(y_f);
    Tensor result{nil, output_shape, x.dtype};
    result.buffer = [queue_.device newBufferWithLength:result.byte_size() options:MTLResourceStorageModeShared];
    if (!result.buffer) throw std::runtime_error("Metal output allocation failed");
    const float* y_host = y_f.data<float>();
    __fp16* out = static_cast<__fp16*>(result.buffer.contents);
    for (size_t index = 0; index < size_t(rows) * n; ++index) out[index] = __fp16(y_host[index]);
    if (!has_bias) return result;
    return run("bias_add", {result, store.tensor(prefix + ".bias")}, output_shape, x.dtype,
               ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
                 return [graph additionWithPrimaryTensor:input[0] secondaryTensor:cast_to(graph, input[1], input[0].dataType) name:nil];
               });
  }
#endif
  if ((mode == "native" || mode == "auto") && x.dtype == MPSDataTypeFloat16 &&
      rows >= 512 && k % 8 == 0) {
    Tensor result{nil, output_shape, x.dtype};
    result.buffer = [queue_.device newBufferWithLength:result.byte_size() options:MTLResourceStorageModeShared];
    if (!result.buffer) throw std::runtime_error("Metal output allocation failed");
    @autoreleasepool {
      static id<MTLLibrary> q4_library;
      static dispatch_once_t once_token;
      dispatch_once(&once_token, ^{
        NSError* library_error = nil;
        NSURL* library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:ONYX_Q4_METALLIB]];
        q4_library = [queue_.device newLibraryWithURL:library_url error:&library_error];
        if (!q4_library) throw std::runtime_error("Could not load q4 kernel library: " + std::string(library_error.localizedDescription.UTF8String));
      });
      id<MTLFunction> function = [q4_library newFunctionWithName:@"onyx_q4_gemm_fp16acc32_mma64"];
      NSError* pipeline_error = nil;
      id<MTLComputePipelineState> pipeline = [queue_.device newComputePipelineStateWithFunction:function error:&pipeline_error];
      if (!pipeline) throw std::runtime_error("Could not create q4 mma64 pipeline: " + std::string(pipeline_error.localizedDescription.UTF8String));
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:x.buffer offset:x.byte_offset atIndex:0];
      [encoder setBuffer:weight.buffer offset:weight.byte_offset atIndex:1];
      [encoder setBuffer:scales.buffer offset:scales.byte_offset atIndex:2];
      [encoder setBuffer:result.buffer offset:0 atIndex:3];
      const uint32_t m32 = static_cast<uint32_t>(rows);
      const uint32_t n32 = static_cast<uint32_t>(n);
      const uint32_t k32 = static_cast<uint32_t>(k);
      const uint32_t g32 = static_cast<uint32_t>(group_size);
      [encoder setBytes:&m32 length:sizeof(m32) atIndex:4];
      [encoder setBytes:&n32 length:sizeof(n32) atIndex:5];
      [encoder setBytes:&k32 length:sizeof(k32) atIndex:6];
      [encoder setBytes:&g32 length:sizeof(g32) atIndex:7];
      const MTLSize groups = MTLSizeMake((n32 + 63) / 64, (m32 + 63) / 64, 1);
      [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status == MTLCommandBufferStatusError) {
        throw std::runtime_error("q4 mma64 command failed: " + std::string(command.error.localizedDescription.UTF8String));
      }
    }
    if (!has_bias) return result;
    return run("bias_add", {result, store.tensor(prefix + ".bias")}, output_shape, x.dtype,
               ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
                 return [graph additionWithPrimaryTensor:input[0] secondaryTensor:cast_to(graph, input[1], input[0].dataType) name:nil];
               });
  }

  const std::string zero_key = shape_key(scales);
  Tensor zero;
  auto cached_zero = zero_cache_.find(zero_key);
  if (cached_zero != zero_cache_.end()) {
    zero = cached_zero->second;
  } else {
    const uint64_t zero_count = scales.elements();
    std::vector<uint8_t> zero_bytes((zero_count + 1) / 2, 0x88);
    id<MTLBuffer> zero_buffer = [queue_.device newBufferWithBytes:zero_bytes.data() length:zero_bytes.size() options:MTLResourceStorageModeShared];
    zero = Tensor{zero_buffer, scales.shape, MPSDataTypeUInt4};
    zero_cache_[zero_key] = zero;
  }
  std::vector<Tensor> inputs = {x, weight, scales, zero};
  if (has_bias) inputs.push_back(store.tensor(prefix + ".bias"));
  const char* graph_setting = std::getenv("ONYX_Q4_GRAPH");
  const std::string variant = graph_setting && *graph_setting && x.shape.size() == 2 ? graph_setting : "f32";
  return run("linear:q4:" + std::to_string(group_size) + ":" + std::to_string(has_bias) + ":" + variant,
             inputs, output_shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    // Dequantizing to a 16-bit type gave wrong results on this MPSGraph, so the weights
    // always dequantize to FP32 and then cast.
    MPSGraphTensor* w = [graph dequantizeTensor:input[1] scaleTensor:input[2] zeroPointTensor:input[3] dataType:MPSDataTypeFloat32 name:nil];
    w = cast_to(graph, w, input[0].dataType);
    MPSGraphTensor* y;
    if (variant == "f32-t") {
      // (W x^T)^T = x W^T. The transposes move from the weight matrix to the activations
      // and to the output, which are an order of magnitude smaller.
      MPSGraphTensor* xt = [graph transposeTensor:input[0] permutation:@[@1, @0] name:nil];
      MPSGraphTensor* yt = [graph matrixMultiplicationWithPrimaryTensor:w secondaryTensor:xt name:nil];
      y = [graph transposeTensor:yt permutation:@[@1, @0] name:nil];
    } else {
      w = [graph transposeTensor:w permutation:@[@1, @0] name:nil];
      y = [graph matrixMultiplicationWithPrimaryTensor:input[0] secondaryTensor:w name:nil];
    }
    if (has_bias) {
      MPSGraphTensor* bias = cast_to(graph, input[4], input[0].dataType);
      y = [graph additionWithPrimaryTensor:y secondaryTensor:bias name:nil];
    }
    return y;
  });
}

Tensor GraphSession::matmul(const Tensor& x, const Tensor& weight) {
  auto output_shape = x.shape;
  output_shape.back() = weight.shape[0];
  return run("matmul", {x, weight}, output_shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* w = cast_to(graph, input[1], input[0].dataType);
    w = [graph transposeTensor:w permutation:@[@1, @0] name:nil];
    return [graph matrixMultiplicationWithPrimaryTensor:input[0] secondaryTensor:w name:nil];
  });
}

Tensor GraphSession::rms_norm(const Tensor& x, const Tensor& weight, double epsilon) {
  return run("rms:" + std::to_string(epsilon), {x, weight}, x.shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* y = rms_unit(graph, input[0], epsilon);
    MPSGraphTensor* w = cast_to(graph, input[1], input[0].dataType);
    return [graph multiplicationWithPrimaryTensor:y secondaryTensor:w name:nil];
  });
}

Tensor GraphSession::modulated_rms_norm(const Tensor& x, const Tensor& weight, const Tensor& scale, const Tensor& shift, double epsilon) {
  return run("rms-modulated:" + std::to_string(epsilon), {x, weight, scale, shift}, x.shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    return modulated_norm(graph, input[0], input[1], input[2], input[3], epsilon);
  });
}

Tensor GraphSession::indexed_modulated_rms_norm(const Tensor& x, const Tensor& weight, const Tensor& table,
                                                const Tensor& indices, uint32_t shift_part, double epsilon) {
  const std::string key = "rms-modulated-indexed:" + std::to_string(shift_part) + ":" + std::to_string(epsilon);
  return run(key, {x, weight, table, indices}, x.shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* base = modulation_base(graph, input[3]);
    MPSGraphTensor* scale = modulation_part(graph, input[2], base, shift_part + 1);
    MPSGraphTensor* shift = modulation_part(graph, input[2], base, shift_part);
    return modulated_norm(graph, input[0], input[1], scale, shift, epsilon);
  });
}

Tensor GraphSession::layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, double epsilon) {
  return run("layer_norm:" + std::to_string(epsilon), {x, weight, bias}, x.shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    const NSInteger axis = static_cast<NSInteger>(x.shape.size()) - 1;
    MPSGraphTensor* mean = [graph meanOfTensor:input[0] axes:@[@(axis)] name:nil];
    MPSGraphTensor* variance = [graph varianceOfTensor:input[0] meanTensor:mean axes:@[@(axis)] name:nil];
    MPSGraphTensor* gamma = cast_to(graph, input[1], input[0].dataType);
    MPSGraphTensor* beta = cast_to(graph, input[2], input[0].dataType);
    return [graph normalizationWithTensor:input[0] meanTensor:mean varianceTensor:variance gammaTensor:gamma betaTensor:beta epsilon:epsilon name:nil];
  });
}

Tensor GraphSession::rope(const Tensor& x, const Tensor& cosine, const Tensor& sine) {
  return run("rope", {x, cosine, sine}, x.shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    return apply_rope(graph, input[0], input[1], input[2]);
  });
}

Tensor GraphSession::attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal, double scale) {
  return run("attention:" + std::to_string(causal) + ":" + std::to_string(scale), {q, k, v}, q.shape, q.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    return scaled_attention(graph, input[0], input[1], input[2], causal, scale);
  });
}

Tensor GraphSession::tile_attention(const Tensor& qkv, const Tensor& cosine, const Tensor& sine) {
  // Q, K and V are contiguous within each head. Keep tiles on the batch axis.
  const int64_t batch = qkv.shape[0];
  const int64_t tokens = qkv.shape[1];
  const int64_t head_dim = 64;
  const int64_t dim = qkv.shape[2] * head_dim;
  return run("vae-tile-attention", {qkv, cosine, sine}, {batch, tokens, dim}, qkv.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* heads = [graph transposeTensor:input[0] permutation:@[@0, @2, @1, @3] name:nil];
    MPSGraphTensor* q = [graph sliceTensor:heads dimension:3 start:0 length:head_dim name:nil];
    MPSGraphTensor* k = [graph sliceTensor:heads dimension:3 start:head_dim length:head_dim name:nil];
    MPSGraphTensor* v = [graph sliceTensor:heads dimension:3 start:2 * head_dim length:head_dim name:nil];
    q = apply_rope(graph, rms_unit(graph, q, 1e-5), input[1], input[2]);
    k = apply_rope(graph, rms_unit(graph, k, 1e-5), input[1], input[2]);
    MPSGraphTensor* result = scaled_attention(graph, q, k, v, false, 0.125f);
    result = [graph transposeTensor:result permutation:@[@0, @2, @1, @3] name:nil];
    return [graph reshapeTensor:result withShape:@[@(batch), @(tokens), @(dim)] name:nil];
  });
}

Tensor GraphSession::fused_qkv_attention(const Tensor& qkv, const Tensor& q_weight, const Tensor& k_weight, const Tensor* cosine, const Tensor* sine) {
  const int64_t rows = qkv.shape[0];
  std::vector<Tensor> inputs = {qkv, q_weight, k_weight};
  if (cosine && sine) {
    inputs.push_back(*cosine);
    inputs.push_back(*sine);
  }
  return run("fused-qkv-attention:" + std::to_string(cosine != nullptr), inputs, {rows, 7168}, qkv.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* fused = [graph reshapeTensor:input[0] withShape:@[@(rows), @56, @3, @128] name:nil];
    fused = [graph transposeTensor:fused permutation:@[@0, @2, @1, @3] name:nil];
    MPSGraphTensor* parts[3];
    for (uint32_t index = 0; index < 3; ++index) {
      MPSGraphTensor* part = [graph sliceTensor:fused dimension:1 start:index length:1 name:nil];
      part = [graph reshapeTensor:part withShape:@[@1, @(rows), @56, @128] name:nil];
      parts[index] = [graph transposeTensor:part permutation:@[@0, @2, @1, @3] name:nil];
    }
    auto normalize = ^MPSGraphTensor*(MPSGraphTensor* value, MPSGraphTensor* weight) {
      MPSGraphTensor* square = [graph squareWithTensor:value name:nil];
      MPSGraphTensor* mean = [graph meanOfTensor:square axes:@[@3] name:nil];
      MPSGraphTensor* epsilon = [graph constantWithScalar:1e-5 shape:@[@1] dataType:value.dataType];
      MPSGraphTensor* sum = [graph additionWithPrimaryTensor:mean secondaryTensor:epsilon name:nil];
      MPSGraphTensor* inverse = [graph reciprocalSquareRootWithTensor:sum name:nil];
      MPSGraphTensor* normalized = [graph multiplicationWithPrimaryTensor:value secondaryTensor:inverse name:nil];
      return [graph multiplicationWithPrimaryTensor:normalized secondaryTensor:weight name:nil];
    };
    parts[0] = normalize(parts[0], input[1]);
    parts[1] = normalize(parts[1], input[2]);
    if (cosine && sine) {
      for (uint32_t index = 0; index < 2; ++index) {
        MPSGraphTensor* first = [graph sliceTensor:parts[index] dimension:3 start:0 length:48 name:nil];
        MPSGraphTensor* second = [graph sliceTensor:parts[index] dimension:3 start:48 length:48 name:nil];
        MPSGraphTensor* negative = [graph negativeWithTensor:second name:nil];
        MPSGraphTensor* rotated = [graph concatTensors:@[negative, first] dimension:3 interleave:NO name:nil];
        MPSGraphTensor* original = [graph concatTensors:@[first, second] dimension:3 interleave:NO name:nil];
        MPSGraphTensor* a = [graph multiplicationWithPrimaryTensor:original secondaryTensor:input[3] name:nil];
        MPSGraphTensor* b = [graph multiplicationWithPrimaryTensor:rotated secondaryTensor:input[4] name:nil];
        MPSGraphTensor* result = [graph additionWithPrimaryTensor:a secondaryTensor:b name:nil];
        MPSGraphTensor* pass = [graph sliceTensor:parts[index] dimension:3 start:96 length:32 name:nil];
        parts[index] = [graph concatTensors:@[result, pass] dimension:3 interleave:NO name:nil];
      }
    }
    MPSGraphTensor* result = scaled_attention(graph, parts[0], parts[1], parts[2], false, 1.0 / std::sqrt(128.0));
    result = [graph transposeTensor:result permutation:@[@0, @2, @1, @3] name:nil];
    return [graph reshapeTensor:result withShape:@[@(rows), @7168] name:nil];
  });
}

Tensor GraphSession::reshape(const Tensor& x, const std::vector<int64_t>& shape) {
  uint64_t count = 1;
  for (int64_t dimension : shape) {
    if (dimension <= 0) throw std::runtime_error("Reshape dimensions must be positive");
    count *= dimension;
  }
  if (count != x.elements()) throw std::runtime_error("Incompatible reshape");
  Tensor result = x;
  result.shape = shape;
  return result;
}

Tensor GraphSession::transpose(const Tensor& x, const std::vector<uint32_t>& permutation) {
  auto output_shape = x.shape;
  for (size_t index = 0; index < permutation.size(); ++index) output_shape[index] = x.shape[permutation[index]];
  NSMutableArray* order = [NSMutableArray new];
  for (uint32_t value : permutation) [order addObject:@(value)];
  return run("transpose", {x}, output_shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    return [graph transposeTensor:input[0] permutation:order name:nil];
  });
}

Tensor GraphSession::slice(const Tensor& x, uint32_t dimension, int64_t start, int64_t length) {
  if (dimension >= x.shape.size() || start < 0 || length <= 0 || start + length > x.shape[dimension]) {
    throw std::runtime_error("Invalid slice");
  }
  auto output_shape = x.shape;
  output_shape[dimension] = length;
  if (dimension == 0) {
    Tensor result = x;
    result.shape = output_shape;
    result.byte_offset += x.byte_size() / x.shape[0] * start;
    return result;
  }
  return run("slice:" + std::to_string(dimension) + ":" + std::to_string(start) + ":" + std::to_string(length),
             {x}, output_shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    return [graph sliceTensor:input[0] dimension:dimension start:start length:length name:nil];
  });
}

Tensor GraphSession::concat(const std::vector<Tensor>& values, uint32_t dimension) {
  auto output_shape = values[0].shape;
  output_shape[dimension] = 0;
  for (const Tensor& value : values) output_shape[dimension] += value.shape[dimension];
  return run("concat:" + std::to_string(dimension), values, output_shape, values[0].dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    return [graph concatTensors:input dimension:dimension interleave:NO name:nil];
  });
}

Tensor GraphSession::add(const Tensor& x, const Tensor& y) {
  return run("add", {x, y}, broadcast_shape(x, y), x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* second = cast_to(graph, input[1], input[0].dataType);
    return [graph additionWithPrimaryTensor:input[0] secondaryTensor:second name:nil];
  });
}

Tensor GraphSession::multiply(const Tensor& x, const Tensor& y) {
  return run("multiply", {x, y}, broadcast_shape(x, y), x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* second = cast_to(graph, input[1], input[0].dataType);
    return [graph multiplicationWithPrimaryTensor:input[0] secondaryTensor:second name:nil];
  });
}

Tensor GraphSession::silu(const Tensor& x) {
  return run("silu", {x}, x.shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* sigmoid = [graph sigmoidWithTensor:input[0] name:nil];
    return [graph multiplicationWithPrimaryTensor:input[0] secondaryTensor:sigmoid name:nil];
  });
}

Tensor GraphSession::gated_activation(const Tensor& x) {
  auto output_shape = x.shape;
  const uint32_t axis = x.shape.size() - 1;
  output_shape[axis] /= 2;
  return run("gated-activation", {x}, output_shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    const int64_t width = input[0].shape[axis].longLongValue;
    MPSGraphTensor* gate = [graph sliceTensor:input[0] dimension:axis start:0 length:width / 2 name:nil];
    MPSGraphTensor* up = [graph sliceTensor:input[0] dimension:axis start:width / 2 length:width / 2 name:nil];
    MPSGraphTensor* activated = [graph sigmoidWithTensor:gate name:nil];
    MPSGraphTensor* gated = [graph multiplicationWithPrimaryTensor:gate secondaryTensor:activated name:nil];
    return [graph multiplicationWithPrimaryTensor:gated secondaryTensor:up name:nil];
  });
}

Tensor GraphSession::gated_residual(const Tensor& x, const Tensor& value, const Tensor& gate) {
  return run("gated-residual", {x, value, gate}, x.shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    return residual_with_gate(graph, input[0], input[1], input[2]);
  });
}

Tensor GraphSession::indexed_gated_residual(const Tensor& x, const Tensor& value, const Tensor& table,
                                           const Tensor& indices, uint32_t gate_part) {
  return run("gated-residual-indexed:" + std::to_string(gate_part), {x, value, table, indices}, x.shape, x.dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    MPSGraphTensor* base = modulation_base(graph, input[3]);
    MPSGraphTensor* gate = modulation_part(graph, input[2], base, gate_part);
    return residual_with_gate(graph, input[0], input[1], gate);
  });
}

Tensor GraphSession::cast(const Tensor& x, MPSDataType dtype) {
  if (x.dtype == dtype) return x;
  return run("cast:" + std::to_string(dtype), {x}, x.shape, dtype,
             ^MPSGraphTensor*(MPSGraph* graph, NSArray<MPSGraphTensor*>* input) {
    return [graph castTensor:input[0] toType:dtype name:nil];
  });
}

std::string GraphSession::profile_json() const {
  using Entry = std::pair<std::string, std::pair<uint64_t, uint64_t>>;
  std::vector<Entry> entries(profile_.begin(), profile_.end());
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
  std::ostringstream output;
  output << "{\"operations\":[";
  for (size_t index = 0; index < entries.size(); ++index) {
    if (index) output << ',';
    output << "{\"name\":\"" << entries[index].first << "\",\"count\":" << entries[index].second.first
           << ",\"nanoseconds\":" << entries[index].second.second << '}';
  }
  output << "]}";
  return output.str();
}

}  // namespace onyx
