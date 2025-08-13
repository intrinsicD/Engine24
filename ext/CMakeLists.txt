target_include_directories(Engine24 SYSTEM PUBLIC glad/include)
target_include_directories(Engine24 SYSTEM PUBLIC imgui)
target_include_directories(Engine24 SYSTEM PUBLIC ImGuizmo)
target_include_directories(Engine24 SYSTEM PUBLIC nanoflann/include)
target_include_directories(Engine24 SYSTEM PUBLIC lbvh)
target_include_directories(Engine24 SYSTEM PUBLIC happly)
target_include_directories(Engine24 SYSTEM PUBLIC glm)

if (NOT TARGET fast_float)
    add_subdirectory(fast_float)
endif ()
target_link_libraries(Engine24 PUBLIC fast_float)

if (NOT TARGET Eigen3::Eigen)
    add_subdirectory(eigen)
endif ()
target_link_libraries(Engine24 PUBLIC Eigen3::Eigen)

if (NOT TARGET EnTT::EnTT)
    add_subdirectory(entt)
endif ()
target_link_libraries(Engine24 PUBLIC EnTT::EnTT)

if (NOT TARGET tinyobjloader)
    add_subdirectory(tinyobjloader)
endif ()
target_link_libraries(Engine24 PRIVATE tinyobjloader)

if (NOT TARGET fmt::fmt)
    add_subdirectory(fmt)
endif ()
target_link_libraries(Engine24 PUBLIC fmt::fmt)

if (NOT TARGET slang)
    add_subdirectory(slang)
endif ()
target_link_libraries(Engine24 PUBLIC slang)

if (NOT TARGET glfw)
    set(GLFW_BUILD_WAYLAND OFF)
    add_subdirectory(glfw)
endif ()
target_link_libraries(Engine24 PRIVATE glfw)

install(TARGETS eigen EXPORT Eigen_export)
install(EXPORT Eigen_export DESTINATION lib/cmake/eigen3)

if (NOT TARGET spectra)
    add_subdirectory(spectra)
endif ()
target_link_libraries(Engine24 PUBLIC Spectra)

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

if (NOT TARGET implot)
    add_library(implot STATIC
            implot/implot.cpp
            implot/implot_demo.cpp
            implot/implot_items.cpp
    )

    target_include_directories(implot PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
endif ()

target_link_libraries(Engine24 PUBLIC bcg_imgui ImGuiFileDialog imguizmo implot)



