cmake_minimum_required(VERSION 3.27)
project(Engine24)

set(CMAKE_CXX_STANDARD 20)

add_library(Engine24 main.cpp
        MeshCompute.cpp
        GLFWUtils.cpp
        GLUtils.cpp
        Mesh.cpp
        PluginTransform.cpp
        PluginCamera.cpp
        PluginMesh.cpp
        PluginOpenGLRenderer.cpp
        FrustumCulling.cpp
)

set(IMGUI_FONTS_PATH "../ext/imgui/misc/fonts")
set(IMGUI_FONTS_PATH_APPS "../../../ext/imgui/misc/fonts")
configure_file(config.h.in config.h @ONLY)
target_include_directories(Engine24 PRIVATE ${CMAKE_BINARY_DIR})

add_executable(Engine24Main main.cpp)
target_link_libraries(Engine24Main PUBLIC Engine24)

add_subdirectory(apps)
add_subdirectory(ext)
add_subdirectory(Core)
add_subdirectory(Primitives)
add_subdirectory(SurfaceMesh)
