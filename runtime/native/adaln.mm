#import "native.hpp"

#include <chrono>
#include <limits>
#include <stdexcept>

namespace onyx {
namespace {

struct GateParameters {
  uint32_t vectors_per_row;
  uint32_t gate_part;
};

id<MTLComputePipelineState> load_gate(id<MTLDevice> device) {
  NSError* error = nil;
  NSURL* url = [NSURL fileURLWithPath:@ONYX_Q4_METALLIB];
  id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
  if (!library) {
    throw std::runtime_error(std::string("Could not load OnyxAdaLN metallib: ") +
                             (error.localizedDescription.UTF8String ?: "unknown error"));
  }
  id<MTLFunction> function = [library newFunctionWithName:@"onyx_adaln_gate"];
  if (!function) throw std::runtime_error("OnyxAdaLN gate kernel not found");
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) {
    throw std::runtime_error(std::string("Could not create OnyxAdaLN gate pipeline: ") +
                             (error.localizedDescription.UTF8String ?: "unknown error"));
  }
  return pipeline;
}

}  // namespace

void GraphSession::metal_gated_residual(Tensor& residual, const Tensor& value, const Tensor& table,
                                      const Tensor& indices, uint32_t gate_part) {
  if (!residual.buffer || !value.buffer || !table.buffer || !indices.buffer ||
      residual.shape.size() != 2 || residual.shape != value.shape ||
      residual.dtype != MPSDataTypeBFloat16 || value.dtype != residual.dtype || table.dtype != residual.dtype ||
      indices.dtype != MPSDataTypeInt32 || indices.shape.size() != 1) {
    throw std::runtime_error("Metal AdaLN gate requires BF16 matrices and int32 modality indices");
  }
  const int64_t rows = residual.shape[0], width = residual.shape[1];
  const bool table_shape_ok = table.shape.size() == 2 && table.shape[1] == width &&
      (table.shape[0] == 12 || table.shape[0] == 18);
  if (rows <= 0 || width <= 0 || width % 4 || indices.shape[0] != rows || !table_shape_ok ||
      (gate_part != 2 && gate_part != 5) || residual.elements() / 4 > std::numeric_limits<uint32_t>::max() ||
      residual.byte_offset % 8 || value.byte_offset % 8 || table.byte_offset % 8 || indices.byte_offset % 4 ||
      residual.buffer == value.buffer || residual.buffer == table.buffer || residual.buffer == indices.buffer) {
    throw std::runtime_error("Metal AdaLN gate has an incompatible tensor layout");
  }

  const auto start = std::chrono::steady_clock::now();
  if (!adaln_gate_pipeline_) adaln_gate_pipeline_ = load_gate(queue_.device);
  const GateParameters params{static_cast<uint32_t>(width / 4), gate_part};
  id<MTLCommandBuffer> command = [queue_ commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:adaln_gate_pipeline_];
  [encoder setBuffer:residual.buffer offset:residual.byte_offset atIndex:0];
  [encoder setBuffer:value.buffer offset:value.byte_offset atIndex:1];
  [encoder setBuffer:table.buffer offset:table.byte_offset atIndex:2];
  [encoder setBuffer:indices.buffer offset:indices.byte_offset atIndex:3];
  [encoder setBytes:&params length:sizeof(params) atIndex:4];
  [encoder dispatchThreads:MTLSizeMake(params.vectors_per_row, rows, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) {
    throw std::runtime_error(std::string("Metal AdaLN gate command failed: ") +
                             (command.error.localizedDescription.UTF8String ?: "unknown error"));
  }
  if (profile_enabled_) {
    auto& timing = profile_["gated-residual-metal"];
    ++timing.first;
    timing.second += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
  }
}

}  // namespace onyx
