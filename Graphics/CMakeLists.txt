target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        Buffer.cpp
        Shader.cpp
        VertexArrayObject.cpp
        GLFWUtils.cpp
        GLUtils.cpp
        Graphics.cpp
        GuiUtils.cpp
        HandleGlfwKeyEvents.cpp
)
