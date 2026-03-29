cmake_minimum_required(VERSION 3.16)
project(gambletron_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(gambletron_engine
    src/engine/hand_eval.cpp
    src/engine/mccfr.cpp
    src/engine/bindings.cpp
)

target_include_directories(gambletron_engine PRIVATE src/engine)

# Optimize for performance
target_compile_options(gambletron_engine PRIVATE -O3 -march=native)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(gambletron_engine PRIVATE Threads::Threads)
