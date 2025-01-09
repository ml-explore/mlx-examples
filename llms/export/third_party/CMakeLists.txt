include(FetchContent)

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_include_directories(
  mlxlm PRIVATE $<BUILD_INTERFACE:${json_SOURCE_DIR}/single_include/nlohmann>)

add_custom_target(unicode)
add_custom_command(
  TARGET unicode 
  PRE_BUILD 
  COMMAND /bin/bash ${CMAKE_CURRENT_SOURCE_DIR}/download_unicode.sh)
add_dependencies(mlxlm unicode)

target_sources(mlxlm
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/unicode.cpp 
  ${CMAKE_CURRENT_BINARY_DIR}/unicode-data.cpp)

target_include_directories(mlxlm PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/..)
