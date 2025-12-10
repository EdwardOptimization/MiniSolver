set(GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

add_executable(car_demo main.cpp)
target_include_directories(car_demo PRIVATE ${GENERATED_DIR})
target_link_libraries(car_demo PRIVATE minisolver_lib)
