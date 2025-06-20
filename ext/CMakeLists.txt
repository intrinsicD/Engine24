target_include_directories(Engine24 SYSTEM PUBLIC glad/include)
target_include_directories(Engine24 SYSTEM PUBLIC imgui)
target_include_directories(Engine24 SYSTEM PUBLIC ImGuizmo)
target_include_directories(Engine24 SYSTEM PUBLIC nanoflann/include)
target_include_directories(Engine24 SYSTEM PUBLIC lbvh)
target_include_directories(Engine24 SYSTEM PUBLIC happly)
target_include_directories(Engine24 SYSTEM PUBLIC glm)

set(GLFW_BUILD_WAYLAND OFF)
add_subdirectory(glfw)
target_link_libraries(Engine24 PRIVATE glfw)

add_subdirectory(fast_float)
target_link_libraries(Engine24 PUBLIC fast_float)

add_subdirectory(eigen)
target_link_libraries(Engine24 PUBLIC Eigen3::Eigen)

add_subdirectory(entt)
target_link_libraries(Engine24 PUBLIC EnTT::EnTT)

add_subdirectory(tinyobjloader)
target_link_libraries(Engine24 PRIVATE tinyobjloader)

add_subdirectory(fmt)
target_link_libraries(Engine24 PUBLIC fmt::fmt)

add_subdirectory(slang)
target_link_libraries(Engine24 PUBLIC slang)

target_sources(Engine24 PRIVATE glad/src/gl.c)

target_include_directories(Engine24 PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends
        ${CMAKE_CURRENT_SOURCE_DIR}/ImGuiFileDialog
        ${CMAKE_CURRENT_SOURCE_DIR}/fmt/include
        ${CMAKE_CURRENT_SOURCE_DIR}/eigen
        ${CMAKE_CURRENT_SOURCE_DIR}/entt/src
        .
)

add_library(bcg_imgui STATIC
        imgui/imgui.cpp
        imgui/imgui_demo.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_tables.cpp
        imgui/misc/cpp/imgui_stdlib.cpp
        imgui/imgui_widgets.cpp
        imgui/backends/imgui_impl_glfw.cpp
        imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(bcg_imgui PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include)

add_library(ImGuiFileDialog STATIC
        ImGuiFileDialog/ImGuiFileDialog.cpp
        ImGuiFileDialog/ImGuiFileDialog.h
        ImGuiFileDialog/ImGuiFileDialogConfig.h
)

target_include_directories(ImGuiFileDialog PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui
        ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include)

add_library(imguizmo STATIC
        ImGuizmo/ImGuizmo.cpp
)

target_include_directories(imguizmo PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

add_library(implot STATIC
        implot/implot.cpp
        implot/implot_demo.cpp
        implot/implot_items.cpp
)

target_include_directories(implot PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/imgui)

target_link_libraries(Engine24 PUBLIC bcg_imgui ImGuiFileDialog imguizmo implot)



