cmake_minimum_required(VERSION 3.27)
project(Engine24 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(Engine24)

include(${CMAKE_SOURCE_DIR}/CopyShaders.cmake)

# Shader directory
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/Graphics/glsl)
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/Shaders/glsl)

# Create the output directory
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# Copy all shader files if not present
copy_shaders(${SHADER_DIR} ${SHADER_OUTPUT_DIR})

set(FONTS_DIR ${CMAKE_SOURCE_DIR}/ext/imgui/misc/fonts)
set(FONTS_OUTPUT_DIR ${CMAKE_BINARY_DIR}/Fonts)

file(MAKE_DIRECTORY ${FONTS_OUTPUT_DIR})

# Copy all shader files if not present
copy_shaders(${FONTS_DIR} ${FONTS_OUTPUT_DIR})

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(ext)

add_subdirectory(Apps)
add_subdirectory(Cuda)
add_subdirectory(Graphics)
add_subdirectory(PointCloud)
add_subdirectory(Primitives)
add_subdirectory(SurfaceMesh)
