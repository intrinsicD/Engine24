cmake_minimum_required(VERSION 3.27)
project(Engine24 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

# Set the output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(Engine24
        include/Guis/MouseGui.h
        src/Guis/MouseGui.cpp
        src/Core/Timer.cpp
        include/Plugins/PluginPointCloudSampling.h
        src/Plugins/PluginPointCloudSampling.cpp
        include/Core/Traits/StringTraits.h
        include/Core/Traits/DimTraits.h
        include/Commands/PointCloudVertexColors.h
        src/Commands/PointCloudVertexColors.cpp
        include/Core/Pool/PoolHandle.h
        include/Core/Geometry/AABB/AABBUtils.h
        include/Core/Traits/VecTraits.h
        include/Core/Module/Module.h
        include/Core/Module/OpenGLRenderModule.h
        include/Core/Geometry/Mesh/SurfaceMeshVertexNormals.h
        src/Core/Geometry/Mesh/SurfaceMeshVertexNormals.cpp
        include/Core/Geometry/Mesh/SurfaceMeshFaceNormals.h
)

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

target_include_directories(Engine24 PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(Engine24 PUBLIC .)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(ext)

add_subdirectory(Apps)
add_subdirectory(Graphics)
add_subdirectory(Cuda)
