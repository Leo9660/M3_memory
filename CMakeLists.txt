cmake_minimum_required(VERSION 3.18)

# ---- CUDA option ----
# Enable with: cmake -DM3_USE_CUDA=ON ..
option(M3_USE_CUDA "Enable CUDA GPU split kernel (requires nvcc)" OFF)

if(M3_USE_CUDA)
    project(agentmemory_m3 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    # Target sm_70 (V100), sm_80 (A100), sm_86 (RTX 30xx) by default;
    # override with -DCMAKE_CUDA_ARCHITECTURES="70;80;86" at configure time.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80 86)
    endif()
    add_compile_definitions(HAVE_CUDA)
    # v1: original global-atomicAdd accumulate_kernel
    set(M3_SPLIT_SRC AgentMemory/M3/src/gpu_split_kernel.cu)
    # v2: shared-memory accumulate_kernel_v2 (256× fewer global atomics)
    set(M3_SPLIT_V2_SRC AgentMemory/M3/src/split_kernel_v2.cu)
    # v3: warp-shuffle accumulate_kernel_v3 (32× fewer shared atomics vs v2)
    #     used by GpuCoordinator::split_gpu_cluster()
    set(M3_SPLIT_V3_SRC AgentMemory/M3/src/split_kernel_v3.cu)
    # v4: 1-warp/vector coalesced reads, float4 loads, on-GPU centroid update,
    #     GPU-side convergence flag (1 int D2H/iter vs n ints in v1/v2/v3),
    #     device-pointer variant for GPU-resident clusters (skips 16MB H2D)
    set(M3_SPLIT_V4_SRC AgentMemory/M3/src/kmeans_gpu_v4.cu)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "M3: CUDA split kernels enabled — v1 + v2 + v3 + v4 (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    project(agentmemory_m3 LANGUAGES CXX)
    # CPU fallback: gpu_split_kernel_cpu.cpp defines gpu_split_kmeans(),
    # gpu_split_kmeans_v2(), and gpu_split_kmeans_v3() stubs.
    set(M3_SPLIT_SRC AgentMemory/M3/src/gpu_split_kernel_cpu.cpp)
    set(M3_SPLIT_V2_SRC "")
    set(M3_SPLIT_V3_SRC "")
    set(M3_SPLIT_V4_SRC "")
    message(STATUS "M3: CPU split kernel (no CUDA)")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(M3_SRC
    AgentMemory/M3/src/bindings_async.cpp
    AgentMemory/M3/src/m3_async.cpp
    AgentMemory/M3/src/m3_index.cpp
    AgentMemory/M3/src/m3_multi_level.cpp
    AgentMemory/M3/src/cluster.cpp
    AgentMemory/M3/src/base.cpp
    AgentMemory/M3/src/gpu_budget.cpp
    AgentMemory/M3/src/gpu_insert_buffer.cpp
    AgentMemory/M3/src/gpu_cluster_index.cpp
    AgentMemory/M3/src/gpu_flush_coordinator.cpp
    AgentMemory/M3/src/gpu_coordinator.cpp
    AgentMemory/M3/src/m3_logger.cpp
    ${M3_SPLIT_SRC}
    ${M3_SPLIT_V2_SRC}
    ${M3_SPLIT_V3_SRC}
    ${M3_SPLIT_V4_SRC}
)

# Core sources without Python bindings (used by standalone test executables)
set(M3_CORE_SRC
    AgentMemory/M3/src/m3_async.cpp
    AgentMemory/M3/src/m3_index.cpp
    AgentMemory/M3/src/m3_multi_level.cpp
    AgentMemory/M3/src/cluster.cpp
    AgentMemory/M3/src/base.cpp
    AgentMemory/M3/src/gpu_budget.cpp
    AgentMemory/M3/src/gpu_insert_buffer.cpp
    AgentMemory/M3/src/gpu_cluster_index.cpp
    AgentMemory/M3/src/gpu_flush_coordinator.cpp
    AgentMemory/M3/src/gpu_coordinator.cpp
    AgentMemory/M3/src/m3_logger.cpp
    ${M3_SPLIT_SRC}
    ${M3_SPLIT_V2_SRC}
    ${M3_SPLIT_V3_SRC}
    ${M3_SPLIT_V4_SRC}
)

set(M3_INCLUDE
    ${CMAKE_SOURCE_DIR}/AgentMemory/M3/include
)

pybind11_add_module(_m3_async MODULE ${M3_SRC})
target_include_directories(_m3_async PRIVATE ${M3_INCLUDE})
target_compile_features(_m3_async PRIVATE cxx_std_17)
target_compile_options(_m3_async PRIVATE -O3 -pthread)

install(TARGETS _m3_async
    LIBRARY DESTINATION AgentMemory/M3
)

# ---- Standalone test: GPU coordination blocks 1-9 ----
add_executable(test_gpu_coord
    test/test_gpu_coord_blocks123.cpp
    ${M3_CORE_SRC}
)
target_include_directories(test_gpu_coord PRIVATE ${M3_INCLUDE})
target_compile_features(test_gpu_coord PRIVATE cxx_std_17)
target_compile_options(test_gpu_coord PRIVATE -O1 -g -pthread)
target_link_libraries(test_gpu_coord PRIVATE pthread)

# ---- Benchmark: GPU split path vs L2-export path ----
# Measures the latency saved by reading cluster data from GPU-resident storage
# (export_cluster) versus exporting from the L2 IVF index (export_l2_cluster)
# before running the k-means split kernel.
#
# Usage: ./benchmark_gpu_split [n_vectors] [dim] [n_trials] [max_iters]
add_executable(benchmark_gpu_split
    test/benchmark_gpu_split.cpp
    ${M3_CORE_SRC}
)
target_include_directories(benchmark_gpu_split PRIVATE ${M3_INCLUDE})
target_compile_features(benchmark_gpu_split PRIVATE cxx_std_17)
target_compile_options(benchmark_gpu_split PRIVATE -O3 -pthread)
target_link_libraries(benchmark_gpu_split PRIVATE pthread)
if(M3_USE_CUDA)
    target_link_libraries(benchmark_gpu_split PRIVATE CUDA::cudart)
endif()

# ---- Benchmark: v1 global-atomic vs v2 shared-mem accumulate kernel ----
# Isolates the accumulate phase to show global-memory read stall in v1.
# Both kernels read the same bytes; v1 is stalled by n×dim global atomics.
# Usage: ./bench_kernel_comparison
if(M3_USE_CUDA)
    add_executable(bench_kernel_comparison
        test/bench_kernel_comparison.cu
    )
    target_include_directories(bench_kernel_comparison PRIVATE ${M3_INCLUDE})
    set_property(TARGET bench_kernel_comparison PROPERTY CUDA_STANDARD 17)
    target_compile_options(bench_kernel_comparison PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_link_libraries(bench_kernel_comparison PRIVATE CUDA::cudart)
    message(STATUS "M3: bench_kernel_comparison added (requires M3_USE_CUDA=ON)")
endif()

# ---- Benchmark: v1/v2-sweep/v3(warp-shfl)/v4(scatter+cuBLAS) ----
# Writes bench_results.csv; plot with: python3 test/plot_bench_results.py
if(M3_USE_CUDA)
    add_executable(bench_kernels_full
        test/bench_kernels_full.cu
    )
    target_include_directories(bench_kernels_full PRIVATE ${M3_INCLUDE})
    set_property(TARGET bench_kernels_full PROPERTY CUDA_STANDARD 17)
    target_compile_options(bench_kernels_full PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_link_libraries(bench_kernels_full PRIVATE CUDA::cudart CUDA::cublas)
    message(STATUS "M3: bench_kernels_full added (requires M3_USE_CUDA=ON)")
endif()

# ---- Benchmark: v3 (reference) vs v4 (warp-coalesced, on-GPU centroid update) ----
# Per-step CUDA event profiling: assign, D2H, accumulate, centroid update.
# Includes full k-means loop comparison and memory bandwidth analysis.
# Usage: ./bench_kmeans_v4
if(M3_USE_CUDA)
    add_executable(bench_kmeans_v4
        test/bench_kmeans_v4.cu
    )
    target_include_directories(bench_kmeans_v4 PRIVATE ${M3_INCLUDE})
    set_property(TARGET bench_kmeans_v4 PROPERTY CUDA_STANDARD 17)
    target_compile_options(bench_kmeans_v4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_link_libraries(bench_kmeans_v4 PRIVATE CUDA::cudart)
    message(STATUS "M3: bench_kmeans_v4 added (requires M3_USE_CUDA=ON)")
endif()
