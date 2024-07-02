cmake_minimum_required(VERSION 3.27)
project(Engine24)

set(CMAKE_CXX_STANDARD 20)

add_executable(Engine24 main.cpp
        MeshCompute.cpp
        GLFWUtils.cpp
        PluginMesh.cpp
        PluginOpenGLRenderer.cpp
        FrustumCulling.cpp
)

add_subdirectory(ext)
add_subdirectory(Core)
add_subdirectory(SurfaceMesh)
