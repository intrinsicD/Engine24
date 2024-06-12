target_include_directories(Engine24 SYSTEM PUBLIC glad/include)
target_include_directories(Engine24 SYSTEM PUBLIC imgui)

add_subdirectory(glfw)
target_sources(Engine24 PRIVATE
        imgui/imgui.cpp
        imgui/misc/cpp/imgui_stdlib.cpp
        imgui/imgui_demo.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_widgets.cpp
        imgui/backends/imgui_impl_glfw.cpp
        imgui/backends/imgui_impl_opengl3.cpp
        ImGuizmo/ImGuizmo.cpp
        implot/implot.cpp
        implot/implot_demo.cpp
        implot/implot_items.cpp)
target_link_libraries(Engine24 glfw)