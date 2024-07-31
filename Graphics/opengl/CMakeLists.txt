target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        Attribute.cpp
        Buffer.cpp
        FrameBuffer.cpp
        GLUtils.cpp
        OpenGLState.cpp
        Program.cpp
        Shader.cpp
        Texture.cpp
        VertexArrayObject.cpp
)