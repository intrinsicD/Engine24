target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        Buffer.cpp
        FrameBuffer.cpp
        GLUtils.cpp
        OpenGLState.cpp
        Program.cpp
        Shader.cpp
        Texture.cpp
        VertexArrayObject.cpp
)