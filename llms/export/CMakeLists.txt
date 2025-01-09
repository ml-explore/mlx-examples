cmake_minimum_required(VERSION 3.27)

project(mlxlm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python 3.9
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_library(mlxlm)
target_link_libraries(mlxlm PUBLIC mlx)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party)

target_sources(mlxlm
  PRIVATE
  mlxlm.cpp
  tokenizer.cpp)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE mlxlm)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE mlxlm)
