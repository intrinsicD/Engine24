target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        GLFWUtils.cpp
        Graphics.cpp
        GuiUtils.cpp
        HandleGlfwKeyEvents.cpp
)

add_subdirectory(opengl)
