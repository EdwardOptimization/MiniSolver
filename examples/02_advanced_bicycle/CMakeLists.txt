set(GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

add_executable(advanced_benchmark advanced_benchmark.cpp)
target_include_directories(advanced_benchmark PRIVATE ${GENERATED_DIR})
target_link_libraries(advanced_benchmark PRIVATE minisolver_lib)

add_executable(advanced_debug advanced_debug.cpp)
target_include_directories(advanced_debug PRIVATE ${GENERATED_DIR})
target_link_libraries(advanced_debug PRIVATE minisolver_lib)