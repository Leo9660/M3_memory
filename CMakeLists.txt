cmake_minimum_required(VERSION 3.18)
project(agentmemory_m3 LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(M3_SRC
    AgentMemory/M3/src/bindings_async.cpp
    AgentMemory/M3/src/m3_async.cpp
    AgentMemory/M3/src/m3_index.cpp
    AgentMemory/M3/src/cluster.cpp
    AgentMemory/M3/src/base.cpp
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