cmake_minimum_required(VERSION 3.25)
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.2" CACHE STRING "Minimum macOS version")
project(onyx_infer VERSION 0.4.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

option(ONYX_ENABLE_MLX "Build the optional MLX Q4 backend when its wheel is available" ON)
set(MLX_PACKAGE_DIR "$ENV{MLX_PACKAGE_DIR}" CACHE PATH "Installed mlx wheel directory")
if(ONYX_ENABLE_MLX AND NOT MLX_PACKAGE_DIR)
  file(GLOB mlx_wheels "${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python*/site-packages/mlx/lib/libmlx.dylib")
  list(LENGTH mlx_wheels mlx_wheel_count)
  if(mlx_wheel_count EQUAL 1)
    get_filename_component(mlx_lib_dir "${mlx_wheels}" DIRECTORY)
    get_filename_component(MLX_PACKAGE_DIR "${mlx_lib_dir}" DIRECTORY)
  endif()
endif()
if(ONYX_ENABLE_MLX AND MLX_PACKAGE_DIR)
  foreach(path IN ITEMS include/mlx/mlx.h lib/libmlx.dylib lib/mlx.metallib)
    if(NOT EXISTS "${MLX_PACKAGE_DIR}/${path}")
      message(FATAL_ERROR "MLX_PACKAGE_DIR is missing ${path}: ${MLX_PACKAGE_DIR}")
    endif()
  endforeach()
  set(ONYX_MLX_AVAILABLE ON)
  message(STATUS "MLX backend enabled: ${MLX_PACKAGE_DIR}")
endif()

set(METAL_COMPILER "$ENV{METAL_COMPILER}" CACHE FILEPATH "Path to the Metal compiler")
if(NOT METAL_COMPILER)
  execute_process(COMMAND xcrun --find metal OUTPUT_VARIABLE METAL_COMPILER
                  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
endif()
get_filename_component(METAL_TOOLCHAIN_BIN "${METAL_COMPILER}" DIRECTORY)
find_program(METALLIB NAMES metallib PATHS "${METAL_TOOLCHAIN_BIN}" NO_DEFAULT_PATH REQUIRED)

if(IS_DIRECTORY "${CMAKE_OSX_SYSROOT}")
  set(MACOS_SDK "${CMAKE_OSX_SYSROOT}")
else()
  execute_process(COMMAND xcrun --sdk macosx --show-sdk-path OUTPUT_VARIABLE MACOS_SDK
                  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
endif()

set(Q4_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/onyx_q4.metallib")
set(METAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ops/metal")
set(MLX_METAL_HEADERS "${METAL_DIR}/mlx_root")
file(GLOB_RECURSE Q4_STEEL_HEADERS CONFIGURE_DEPENDS "${MLX_METAL_HEADERS}/*.h")
# q4_gemm_nax instantiates MLX's neural-accelerator QMM kernel, which includes the Metal
# Performance Primitives headers from the SDK.
foreach(kernel IN ITEMS q4_gemm q4_gemm_simd q4_gemm_steel q4_gemm_nax adaln)
  set(source "${METAL_DIR}/${kernel}.metal")
  set(air "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.air")
  add_custom_command(
    OUTPUT "${air}"
    COMMAND "${METAL_COMPILER}" -std=metal4.0 "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}"
            "-isysroot${MACOS_SDK}" -I "${MLX_METAL_HEADERS}" -c "${source}" -o "${air}"
    DEPENDS "${source}" ${Q4_STEEL_HEADERS}
    VERBATIM
  )
  list(APPEND Q4_AIR "${air}")
endforeach()
add_custom_command(
  OUTPUT "${Q4_LIBRARY}"
  COMMAND "${METALLIB}" ${Q4_AIR} -o "${Q4_LIBRARY}"
  DEPENDS ${Q4_AIR}
  VERBATIM
)
add_custom_target(onyx_metal_library DEPENDS "${Q4_LIBRARY}")

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
find_library(IMAGEIO_FRAMEWORK ImageIO REQUIRED)
find_library(UNIFORM_TYPEIDENTIFIERS_FRAMEWORK UniformTypeIdentifiers REQUIRED)
find_path(MPSCORE_INCLUDE_DIR MPSCore/MPSNDArray.h
          PATHS "${MACOS_SDK}/System/Library/Frameworks/MetalPerformanceShaders.framework/Versions/A/Frameworks"
          NO_DEFAULT_PATH REQUIRED)

add_compile_options(-fobjc-arc -Wall -Wextra -Werror)

add_library(onyx_weight_store STATIC runtime/native/weight_store.mm)
target_include_directories(onyx_weight_store PUBLIC runtime/native)
target_link_libraries(onyx_weight_store PUBLIC "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")

add_library(onyx_graph STATIC runtime/native/graph.mm runtime/native/q4_nax.mm runtime/native/adaln.mm)
add_dependencies(onyx_graph onyx_metal_library)
target_include_directories(onyx_graph SYSTEM PRIVATE "${MPSCORE_INCLUDE_DIR}")
target_compile_definitions(onyx_graph PRIVATE ONYX_Q4_METALLIB="${Q4_LIBRARY}")
target_link_libraries(onyx_graph PUBLIC onyx_weight_store "${MPSGRAPH_FRAMEWORK}" "${MPS_FRAMEWORK}")
if(ONYX_MLX_AVAILABLE)
  target_include_directories(onyx_graph SYSTEM PUBLIC "${MLX_PACKAGE_DIR}/include" "${MLX_PACKAGE_DIR}/include/metal_cpp")
  target_compile_definitions(onyx_graph PUBLIC ONYX_MLX_BACKEND PRIVATE MLX_METALLIB_PATH="${MLX_PACKAGE_DIR}/lib/mlx.metallib")
  target_link_libraries(onyx_graph PUBLIC "${MLX_PACKAGE_DIR}/lib/libmlx.dylib")
endif()

add_executable(onyx-native runtime/native/main.mm runtime/native/text.mm runtime/native/dit.mm runtime/native/vae.mm)
target_link_libraries(onyx-native PRIVATE onyx_graph "${COREGRAPHICS_FRAMEWORK}" "${IMAGEIO_FRAMEWORK}" "${UNIFORM_TYPEIDENTIFIERS_FRAMEWORK}")

add_executable(onyx-metal-bench tools/metal_bench.mm)
add_dependencies(onyx-metal-bench onyx_metal_library)
target_compile_definitions(onyx-metal-bench PRIVATE ONYX_METALLIB_PATH="${Q4_LIBRARY}")
target_link_libraries(onyx-metal-bench PRIVATE "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}")

add_executable(onyx-mpsgraph-q4-probe tools/mpsgraph_q4_probe.mm)
target_link_libraries(onyx-mpsgraph-q4-probe PRIVATE "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}" "${MPSGRAPH_FRAMEWORK}")
add_executable(onyx-steel-real-data-probe tools/steel_real_data_probe.mm)
target_link_libraries(onyx-steel-real-data-probe PRIVATE "${FOUNDATION_FRAMEWORK}" "${METAL_FRAMEWORK}" "${MPSGRAPH_FRAMEWORK}")

foreach(probe IN ITEMS mpsgraph_probe q4_linear_probe q4_path_compare dit_block_profile dit_modulation_probe dit_gate_probe linear_steel_probe)
  string(REPLACE "_" "-" target "onyx-${probe}")
  add_executable(${target} "tools/${probe}.mm")
  target_link_libraries(${target} PRIVATE onyx_graph)
endforeach()
add_executable(onyx-weight-store-cache-test tools/weight_store_cache_test.mm)
target_link_libraries(onyx-weight-store-cache-test PRIVATE onyx_weight_store)
add_executable(onyx-mtl4-counter-probe tools/mtl4_counter_probe.mm)
target_link_libraries(onyx-mtl4-counter-probe PRIVATE onyx_weight_store)

enable_testing()
add_test(NAME onyx_dit_modulation_single COMMAND onyx-dit-modulation-probe 1 --verify-only)
add_test(NAME onyx_dit_modulation_ragged COMMAND onyx-dit-modulation-probe 17 --verify-only)
add_test(NAME onyx_dit_metal_gate_single COMMAND onyx-dit-gate-probe 1 --verify-only)
add_test(NAME onyx_dit_metal_gate_ragged COMMAND onyx-dit-gate-probe 17 --verify-only --references)
add_test(NAME onyx_mpsgraph_bf16 COMMAND onyx-mpsgraph-probe)
add_test(NAME onyx_mpsgraph_q4 COMMAND onyx-mpsgraph-q4-probe)
add_test(NAME onyx_q4_gemm_small COMMAND onyx-metal-bench --m 16 --n 128 --k 256 --group 64 --iterations 5)
add_test(NAME onyx_q4_gemm_tile32 COMMAND onyx-metal-bench --m 16 --n 128 --k 256 --group 64 --iterations 5 --variant tile32)
add_test(NAME onyx_q4_gemm_simd COMMAND onyx-metal-bench --m 32 --n 64 --k 64 --group 64 --iterations 3 --variant simd)
add_test(NAME onyx_q4_gemm_steel COMMAND onyx-metal-bench --m 32 --n 64 --k 64 --group 64 --iterations 3 --variant mlx-steel)
add_test(NAME onyx_q4_gemm_mma_small COMMAND onyx-metal-bench --m 16 --n 128 --k 256 --group 64 --iterations 5 --variant mma)
add_test(NAME onyx_q4_gemm_mma_ungrouped COMMAND onyx-metal-bench --m 16 --n 64 --k 128 --group 32 --iterations 5 --variant mma)
# Ragged shapes (k % 8 != 0, partial tiles): caught the q4-mma-v1-r0 class of
# numeric bug; every kernel variant must stay correct on these.
add_test(NAME onyx_q4_gemm_ragged COMMAND onyx-metal-bench --m 17 --n 33 --k 30 --group 30 --iterations 5)
add_test(NAME onyx_q4_gemm_ragged_mma COMMAND onyx-metal-bench --m 17 --n 33 --k 30 --group 30 --iterations 5 --variant mma)
add_test(NAME onyx_q4_gemm_ragged_mma64 COMMAND onyx-metal-bench --m 17 --n 33 --k 30 --group 30 --iterations 5 --variant mma64)
add_test(NAME onyx_q4_gemm_ragged_splitk COMMAND onyx-metal-bench --m 17 --n 33 --k 30 --group 30 --iterations 5 --variant splitk4)
add_test(NAME onyx_q4_gemm_splitk_multitile COMMAND onyx-metal-bench --m 32 --n 64 --k 64 --group 64 --iterations 5 --variant splitk4)

add_test(NAME onyx_q4_linear_mpsgraph COMMAND onyx-q4-linear-probe)
set_tests_properties(onyx_q4_linear_mpsgraph PROPERTIES ENVIRONMENT "ONYX_Q4_LINEAR=mpsgraph")
add_test(NAME onyx_q4_linear_dispatch COMMAND onyx-q4-linear-probe)
set_tests_properties(onyx_q4_linear_dispatch PROPERTIES ENVIRONMENT "ONYX_Q4_LINEAR=;ONYX_Q4_BACKEND=native")
# The NAX path needs BF16 activations. On GPUs without neural accelerators the engine falls
# back to MPSGraph and this check exercises that fallback.
add_test(NAME onyx_q4_linear_nax COMMAND onyx-q4-linear-probe)
set_tests_properties(onyx_q4_linear_nax PROPERTIES ENVIRONMENT "ONYX_Q4_LINEAR=nax")
if(ONYX_MLX_AVAILABLE)
  add_test(NAME onyx_q4_linear_mlx COMMAND onyx-q4-linear-probe)
  set_tests_properties(onyx_q4_linear_mlx PROPERTIES ENVIRONMENT "ONYX_Q4_LINEAR=;ONYX_Q4_BACKEND=mlx")
endif()

set(ONYX_Q4_TEST_PACK "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/models/h3-w4-probes/text-q0-qproj.onxpack"
    CACHE FILEPATH "Optional real quantized weight pack for local probes")
if(EXISTS "${ONYX_Q4_TEST_PACK}")
  add_test(NAME onyx_weight_store_cache COMMAND onyx-weight-store-cache-test "${ONYX_Q4_TEST_PACK}" 67108864)
  add_test(NAME onyx_steel_real_data COMMAND onyx-steel-real-data-probe "${Q4_LIBRARY}" "${ONYX_Q4_TEST_PACK}" 32)
  add_test(NAME onyx_linear_steel_probe COMMAND onyx-linear-steel-probe "${Q4_LIBRARY}" "${ONYX_Q4_TEST_PACK}" 69)
endif()
