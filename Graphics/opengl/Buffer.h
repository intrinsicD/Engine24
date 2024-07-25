//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_BUFFER_H
#define ENGINE24_BUFFER_H

#include <unordered_map>
#include <string>
#include <algorithm>

namespace Bcg {
    struct Buffer {
        unsigned int id = -1;
        unsigned int binding_point = -1;

        operator bool() const {
            return id != -1;
        }

        enum Target : unsigned int{
            ARRAY_BUFFER = 0x8892,
            ELEMENT_ARRAY_BUFFER = 0x8893,
            SHADER_STORAGE_BUFFER = 0x90D2,
            UNIFORM_BUFFER = 0x8A11,
            UNDEFINED = 0,
        } target = UNDEFINED;

        enum Usage {
            STREAM_DRAW = 0x88E0,
            STREAM_READ = 0x88E1,
            STREAM_COPY = 0x88E2,
            STATIC_DRAW = 0x88E4,
            STATIC_READ = 0x88E5,
            STATIC_COPY = 0x88E6,
            DYNAMIC_DRAW = 0x88E8,
            DYNAMIC_READ = 0x88E9,
            DYNAMIC_COPY = 0x88EA
        } usage;

        void create();

        void destroy();

        void bind();

        void bind_base(unsigned int index);

        void unbind();

        void buffer_data(const void *data, unsigned int size_bytes, unsigned int usage);

        void buffer_sub_data(const void *data, unsigned int size_bytes, unsigned int offset = 0);

        void get_buffer_sub_data(void *data, unsigned int size_bytes, unsigned int offset = 0);
    };

    struct ArrayBuffer : public Buffer {
        ArrayBuffer();
    };

    struct ElementArrayBuffer : public Buffer {
        ElementArrayBuffer();
    };

    struct ShaderStorageBuffer : public Buffer {
        ShaderStorageBuffer();
    };

    struct UniformBuffer : public Buffer {
        UniformBuffer();
    };

    struct BufferLayout {
        [[nodiscard]] unsigned int total_size_bytes() const {
            unsigned int size_in_bytes = 0;
            for (const auto &item: layout) {
                size_in_bytes += item.second.size_in_bytes;
            }
            return size_in_bytes;
        }

        struct Layout {
            const char *name{};
            unsigned int size_in_bytes = 0;
            unsigned int dims = 0;          //3
            unsigned int size = 0;          //sizeof(float)
            unsigned int normalized = 0;
            unsigned int offset = 0;
            const void *data{};

            [[nodiscard]] unsigned int stride() const {
                return size * dims;
            }
        };

        Layout &get_or_add(const char *name) {
            auto iter = std::find_if(layout.begin(), layout.end(), [name](auto &&item) {
                return item.second.name == name;
            });

            if (iter == layout.end()) {
                Layout item;
                item.name = name;
                item.offset = total_size_bytes();
                return layout.emplace(name, item).first->second;
            }

            return layout[name];
        }

        std::unordered_map<std::string, Layout> layout;
    };
}

#endif //ENGINE24_BUFFER_H
