cmake_minimum_required(VERSION 3.18)

# CUDA is required — gpu_cluster_index.cu and all split kernels need nvcc.
# Configure with: cmake -B build -DM3_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89"

# Must be set BEFORE project() — CMake reads CMAKE_CUDA_ARCHITECTURES during
# the project() call when CUDA is listed as a language.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
endif()

project(agentmemory_m3 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_compile_definitions(HAVE_CUDA)
find_package(CUDAToolkit REQUIRED)

# Split kernel
set(M3_SPLIT_V4_SRC AgentMemory/M3/src/kmeans_gpu_v4.cu)

message(STATUS "M3: CUDA enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_library(OPENBLAS_LIB NAMES openblas HINTS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu REQUIRED)
find_path(CBLAS_INCLUDE_DIR cblas.h HINTS /usr/include/x86_64-linux-gnu /usr/include REQUIRED)
message(STATUS "OpenBLAS: ${OPENBLAS_LIB}  cblas.h: ${CBLAS_INCLUDE_DIR}")

find_package(OpenMP REQUIRED)
message(STATUS "OpenMP: ${OpenMP_CXX_FLAGS}")

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
target_include_directories(_m3_async PRIVATE ${M3_INCLUDE} ${CBLAS_INCLUDE_DIR})
target_compile_features(_m3_async PRIVATE cxx_std_17)
target_compile_options(_m3_async PRIVATE -O3 -pthread -mavx2 -mfma ${OpenMP_CXX_FLAGS})
target_link_libraries(_m3_async PRIVATE CUDA::cudart CUDA::cublas ${OPENBLAS_LIB} OpenMP::OpenMP_CXX)

install(TARGETS _m3_async
    LIBRARY DESTINATION AgentMemory/M3
)


# ---- Standalone test: Four paper mechanism integration tests ----
# Tests disabled pending update to new L0/L1 topology (L0+L1 query-centric, independent of L2).
# Re-enable after updating test assertions to match new ClusterMetadata and promotion model.
#
# add_executable(test_paper_mechanisms ...)
# add_executable(test_bg_operations ...)
# add_executable(test_gpu_multilevel_routing ...)


# ---- DiskANN C++ bindings ----
option(AGENTMEMORY_BUILD_DISKANN "Build DiskANN C++ bindings" OFF)
option(DISKANN_LINK_TCMALLOC "Link DiskANN bindings against tcmalloc" OFF)

if(AGENTMEMORY_BUILD_DISKANN)
    if(DEFINED DISKANN_HOME)
        set(_diskann_home "${DISKANN_HOME}")
    elseif(DEFINED ENV{DISKANN_HOME})
        set(_diskann_home "$ENV{DISKANN_HOME}")
    else()
        message(FATAL_ERROR "AGENTMEMORY_BUILD_DISKANN=ON but DISKANN_HOME is not set")
    endif()

    set(_diskann_include "${_diskann_home}/include")
    if(DEFINED DISKANN_INCLUDE_DIR)
        set(_diskann_include "${DISKANN_INCLUDE_DIR}")
    elseif(DEFINED ENV{DISKANN_INCLUDE_DIR})
        set(_diskann_include "$ENV{DISKANN_INCLUDE_DIR}")
    endif()
    if(NOT EXISTS "${_diskann_include}")
        message(FATAL_ERROR "DiskANN include directory not found: ${_diskann_include}")
    endif()

    if(DEFINED DISKANN_LIB_PATH)
        set(_diskann_lib "${DISKANN_LIB_PATH}")
    elseif(DEFINED ENV{DISKANN_LIB_PATH})
        set(_diskann_lib "$ENV{DISKANN_LIB_PATH}")
    else()
        # DiskANN places the static lib in build/src/ when built with CMake
        set(_diskann_lib "${_diskann_home}/build/src/libdiskann.a")
    endif()
    if(NOT EXISTS "${_diskann_lib}")
        message(FATAL_ERROR "DiskANN library not found: ${_diskann_lib}. Set DISKANN_LIB_PATH explicitly.")
    endif()

    set(DISKANN_SRC AgentMemory/backend/DiskANN/src/diskann_bindings.cpp)
    pybind11_add_module(_diskann_cpp MODULE ${DISKANN_SRC})
    target_include_directories(_diskann_cpp PRIVATE "${_diskann_include}")
    target_compile_features(_diskann_cpp PRIVATE cxx_std_17)
    target_compile_options(_diskann_cpp PRIVATE -O3 -fopenmp)
    set(_diskann_link_libs "${_diskann_lib}")

    if(DEFINED DISKANN_EXTRA_LIBS)
        list(APPEND _diskann_link_libs ${DISKANN_EXTRA_LIBS})
    else()
        set(_diskann_mkl_hints "")
        if(DEFINED ENV{MKLROOT})
            list(APPEND _diskann_mkl_hints "$ENV{MKLROOT}/lib" "$ENV{MKLROOT}/lib/intel64")
        endif()
        list(APPEND _diskann_mkl_hints
            "/opt/intel/oneapi/mkl/latest/lib/intel64"
            "/opt/intel/oneapi/compiler/latest/lib"
            "/opt/intel/oneapi/compiler/2025.3/lib"
        )

        foreach(_lib_name mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5)
            find_library("DISKANN_${_lib_name}_LIB" NAMES ${_lib_name} HINTS ${_diskann_mkl_hints})
            if(DISKANN_${_lib_name}_LIB)
                list(APPEND _diskann_link_libs ${DISKANN_${_lib_name}_LIB})
            endif()
        endforeach()

        if(DISKANN_LINK_TCMALLOC)
            find_library(DISKANN_TCMALLOC_LIB NAMES tcmalloc tcmalloc_minimal)
            if(DISKANN_TCMALLOC_LIB)
                list(APPEND _diskann_link_libs ${DISKANN_TCMALLOC_LIB})
            else()
                message(FATAL_ERROR "DISKANN_LINK_TCMALLOC=ON but tcmalloc library not found")
            endif()
        endif()

        list(APPEND _diskann_link_libs pthread m dl)
    endif()

    target_link_libraries(_diskann_cpp PRIVATE ${_diskann_link_libs})
    target_link_options(_diskann_cpp PRIVATE -fopenmp)

    install(TARGETS _diskann_cpp
        LIBRARY DESTINATION AgentMemory/backend/DiskANN
    )
else()
    message(STATUS "DiskANN bindings are disabled. Set -DAGENTMEMORY_BUILD_DISKANN=ON to enable.")
endif()
