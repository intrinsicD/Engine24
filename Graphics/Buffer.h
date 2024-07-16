//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_BUFFER_H
#define ENGINE24_BUFFER_H

namespace Bcg{
    struct Buffer {
        unsigned int id = -1;
        unsigned int target = -1;

        enum Usage{
            STREAM_DRAW = 0x88E0,
            STREAM_READ = 0x88E1,
            STREAM_COPY = 0x88E2,
            STATIC_DRAW = 0x88E4,
            STATIC_READ = 0x88E5,
            STATIC_COPY = 0x88E6,
            DYNAMIC_DRAW = 0x88E8,
            DYNAMIC_READ = 0x88E9,
            DYNAMIC_COPY = 0x88EA
        };

        void create();

        void destroy();

        void bind();

        void unbind();

        void buffer_data(const void *data, unsigned int size_bytes, unsigned int usage);

        void buffer_sub_data(const void *data, unsigned int size_bytes, unsigned int offset = 0);

        void get_buffer_sub_data(void *data, unsigned int size_bytes, unsigned int offset = 0);
    };

    struct ArrayBuffer : public Buffer{
        ArrayBuffer();
    };

    struct ElementArrayBuffer : public Buffer{
        ElementArrayBuffer();
    };

    struct ShaderStorageBuffer : public Buffer{
        ShaderStorageBuffer();

        void bind_base(unsigned int index);
    };

    struct UniformBuffer : public Buffer{
        UniformBuffer();
    };
}

#endif //ENGINE24_BUFFER_H
