cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(CS636_Final_Project)

set(kernel_source_files
    kernel.cpp
)

add_subdirectory(extern/pybind11)
pybind11_add_module(kernel ${kernel_source_files})
target_link_libraries(kernel PRIVATE)