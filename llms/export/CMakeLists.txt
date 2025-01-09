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

include(FetchContent)

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_include_directories(
  mlxlm PRIVATE $<BUILD_INTERFACE:${json_SOURCE_DIR}/single_include/nlohmann>)

target_sources(mlxlm
  PRIVATE
  mlxlm.cpp
  tokenizer.cpp
  unicode.cpp
  unicode_data.cpp)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE mlxlm)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE mlxlm)
