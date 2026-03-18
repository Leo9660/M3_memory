cmake_minimum_required(VERSION 3.18)

# CUDA is required — gpu_cluster_index.cu and all split kernels need nvcc.
# Configure with: cmake -B build -DM3_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
project(agentmemory_m3 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

# Target sm_80 (A100), sm_86 (RTX 30xx), sm_89 (RTX 40xx) by default.
# Override: cmake -B build -DCMAKE_CUDA_ARCHITECTURES="70;80;86"
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
endif()

add_compile_definitions(HAVE_CUDA)
find_package(CUDAToolkit REQUIRED)

# Split kernel
set(M3_SPLIT_V4_SRC AgentMemory/M3/src/kmeans_gpu_v4.cu)

message(STATUS "M3: CUDA enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")

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
    AgentMemory/M3/src/gpu_cluster_index.cu
    AgentMemory/M3/src/gpu_flush_coordinator.cpp
    AgentMemory/M3/src/gpu_coordinator.cpp
    AgentMemory/M3/src/m3_logger.cpp
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
    AgentMemory/M3/src/gpu_cluster_index.cu
    AgentMemory/M3/src/gpu_flush_coordinator.cpp
    AgentMemory/M3/src/gpu_coordinator.cpp
    AgentMemory/M3/src/m3_logger.cpp
    ${M3_SPLIT_V4_SRC}
)

set(M3_INCLUDE
    ${CMAKE_SOURCE_DIR}/AgentMemory/M3/include
)

pybind11_add_module(_m3_async MODULE ${M3_SRC})
target_include_directories(_m3_async PRIVATE ${M3_INCLUDE})
target_compile_features(_m3_async PRIVATE cxx_std_17)
target_compile_options(_m3_async PRIVATE -O3 -pthread)
target_link_libraries(_m3_async PRIVATE CUDA::cudart)

install(TARGETS _m3_async
    LIBRARY DESTINATION AgentMemory/M3
)


# ---- Standalone test: Four paper mechanism integration tests ----
# Mechanism 1: Hotspot-Aware Caching
# Mechanism 2: CPU Insertion Buffer (B_insert=128)
# Mechanism 3: Async Consistency Management (buffer flush)
# Mechanism 4: On-GPU Cluster Splitting
add_executable(test_paper_mechanisms
    test/test_paper_mechanisms.cpp
    ${M3_CORE_SRC}
)
target_include_directories(test_paper_mechanisms PRIVATE ${M3_INCLUDE})
target_compile_features(test_paper_mechanisms PRIVATE cxx_std_17)
target_compile_options(test_paper_mechanisms PRIVATE -O1 -g -pthread)
target_link_libraries(test_paper_mechanisms PRIVATE pthread CUDA::cudart)

# ---- Standalone test: Background maintenance operations ----
# Tests flush_buffers(), cpu_maintenance(), and rebalance() on GpuCoordinator.
add_executable(test_bg_operations
    test/test_bg_operations.cpp
    ${M3_CORE_SRC}
)
target_include_directories(test_bg_operations PRIVATE ${M3_INCLUDE})
target_compile_features(test_bg_operations PRIVATE cxx_std_17)
target_compile_options(test_bg_operations PRIVATE -O1 -g -pthread)
target_link_libraries(test_bg_operations PRIVATE pthread CUDA::cudart)

# ---- Standalone test: GPU/CPU routing through MultiLevelIndex ----
# Tests set_gpu_coordinator() wiring: insert routing (buffer vs L0+L2),
# search routing (GPU kernel vs CPU L2, mixed probe sets, early termination).
add_executable(test_gpu_multilevel_routing
    test/test_gpu_multilevel_routing.cpp
    ${M3_CORE_SRC}
)
target_include_directories(test_gpu_multilevel_routing PRIVATE ${M3_INCLUDE})
target_compile_features(test_gpu_multilevel_routing PRIVATE cxx_std_17)
target_compile_options(test_gpu_multilevel_routing PRIVATE -O1 -g -pthread)
target_link_libraries(test_gpu_multilevel_routing PRIVATE pthread CUDA::cudart)

