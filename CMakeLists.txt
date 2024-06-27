cmake_minimum_required(VERSION 3.27)
project(Engine24)

set(CMAKE_CXX_STANDARD 20)

add_executable(Engine24 main.cpp
        Engine.cpp
        Graphics.cpp
        Keyboard.cpp
        Mouse.cpp
        FileWatcher.cpp
        MeshCompute.cpp
        GLFWUtils.cpp
        Input.cpp
        Logger.cpp
        Plugins.cpp
        Plugin.cpp
        PluginMesh.cpp
        PluginOpenGLRenderer.cpp
        FrustumCulling.cpp
)

add_subdirectory(ext)
add_subdirectory(SurfaceMesh)
