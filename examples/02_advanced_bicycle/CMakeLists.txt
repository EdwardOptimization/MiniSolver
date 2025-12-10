set(GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

add_executable(bicycle_benchmark main_benchmark.cpp)
target_include_directories(bicycle_benchmark PRIVATE ${GENERATED_DIR})
target_link_libraries(bicycle_benchmark PRIVATE minisolver_lib)
