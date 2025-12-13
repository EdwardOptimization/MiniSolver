cmake_minimum_required(VERSION 3.5)

# Line Search Benchmark Example
add_executable(line_search_benchmark main.cpp)
target_link_libraries(line_search_benchmark minisolver_lib)

# Include generated model headers
target_include_directories(line_search_benchmark PRIVATE generated)
