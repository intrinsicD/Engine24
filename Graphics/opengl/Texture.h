//
// Created by alex on 22.07.24.
//

#ifndef ENGINE24_TEXTURE_H
#define ENGINE24_TEXTURE_H

#include <vector>
#include <queue>
#include <string>

namespace Bcg {
    // Represents an OpenGL texture object and its properties.
    struct Texture {
        enum Type {
            UNSIGNED_BYTE = 0x1401, // GL_UNSIGNED_BYTE
            UNSIGNED_SHORT = 0x1403, // GL_UNSIGNED_SHORT
            UNSIGNED_INT = 0x1405, // GL_UNSIGNED_INT
            FLOAT = 0x1406, // GL_FLOAT
        };

        enum Format {
            RED = 0x1903, // GL_RED
            RG = 0x8227, // GL_RG
            RGB = 0x1907, // GL_RGB
            RGBA = 0x1908, // GL_RGBA
            DEPTH_COMPONENT = 0x1902, // GL_DEPTH_COMPONENT
            DEPTH_STENCIL = 0x84F9, // GL_DEPTH_STENCIL
        };

        enum InternalFormat {
            R8 = 0x8229, // GL_R8
            RG8 = 0x822B, // GL_RG8
            RGB8 = 0x8051, // GL_RGB8
            RGBA8 = 0x8058, // GL_RGBA8
            R16F = 0x822D, // GL_R16F
            RG16F = 0x822F, // GL_RG16F
            RGB16F = 0x881B, // GL_RGB16F
            RGBA16F = 0x881A, // GL_RGBA16F
            DEPTH_COMPONENT24 = 0x81A6, // GL_DEPTH_COMPONENT24
            DEPTH24_STENCIL8 = 0x88F0, // GL_DEPTH24_STENCIL8
        };

        enum Target {
            TEXTURE_1D = 0x0DE0, // GL_TEXTURE_1D
            PROXY_TEXTURE_1D = 0x8063, // GL_PROXY_TEXTURE_1D
            TEXTURE_2D = 0x0DE1, // GL_TEXTURE_2D
            PROXY_TEXTURE_2D = 0x8064, // GL_PROXY_TEXTURE_2D
            TEXTURE_1D_ARRAY = 0x8C18, // GL_TEXTURE_1D_ARRAY
            PROXY_TEXTURE_1D_ARRAY = 0x8C19, // GL_PROXY_TEXTURE_1D_ARRAY
            TEXTURE_RECTANGLE = 0x84F5, // GL_TEXTURE_RECTANGLE
            PROXY_TEXTURE_RECTANGLE = 0x84F7, // GL_PROXY_TEXTURE_RECTANGLE
            TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515, // GL_TEXTURE_CUBE_MAP_POSITIVE_X
            TEXTURE_CUBE_MAP_NEGATIVE_X = 0x8516, // GL_TEXTURE_CUBE_MAP_NEGATIVE_X
            TEXTURE_CUBE_MAP_POSITIVE_Y = 0x8517, // GL_TEXTURE_CUBE_MAP_POSITIVE_Y
            TEXTURE_CUBE_MAP_NEGATIVE_Y = 0x8518, // GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
            TEXTURE_CUBE_MAP_POSITIVE_Z = 0x8519, // GL_TEXTURE_CUBE_MAP_POSITIVE_Z
            TEXTURE_CUBE_MAP_NEGATIVE_Z = 0x851A, // GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
            PROXY_TEXTURE_CUBE_MAP = 0x851B, // GL_PROXY_TEXTURE_CUBE_MAP
            TEXTURE_3D = 0x806F, // GL_TEXTURE_3D
            PROXY_TEXTURE_3D = 0x8070, // GL_PROXY_TEXTURE_3D
            TEXTURE_2D_ARRAY = 0x8C1A, // GL_TEXTURE_2D_ARRAY
            PROXY_TEXTURE_2D_ARRAY = 0x8C1B, // GL_PROXY_TEXTURE_2D_ARRAY
        };

        // things like id, target, width, height, format, internal_format, type etc
        int unit = -1; // Location in the shader (used for binding)
        unsigned int type;
        std::string shader_name;
        std::string bound_buffer_name;

        unsigned int id = static_cast<unsigned int>(-1); // OpenGL texture ID
        unsigned int target;
        unsigned int format;
        unsigned int internal_format;

        unsigned int width = 0;
        unsigned int height = 0;
        unsigned int channels = 0;

        int level = 0; // Mipmap level
        int border = 0; // Border width

        operator bool() const {
            return id != -1;
        }

        // Creates the OpenGL texture (implementation in .cpp)
        void create();

        // Destroys the OpenGL texture (implementation in .cpp)
        void destroy();

        void bind() const;

        void unbind() const;

        void use(unsigned int program_id) const;
    };

    // 1D texture specialization
    struct Texture1D : public Texture {
        enum Targets1D {
            TEXTURE_1D = 0x0DE0, // GL_TEXTURE_1D
            PROXY_TEXTURE_1D = 0x8063, // GL_PROXY_TEXTURE_1D
        };

        void set_data(const void *data, unsigned int width);
    };

    // 2D texture specialization
    struct Texture2D : public Texture {
        enum Targets2D {
            // 2D texture targets
            TEXTURE_2D = 0x0DE1, // GL_TEXTURE_2D
            PROXY_TEXTURE_2D = 0x8064, // GL_PROXY_TEXTURE_2D

            // 1D array texture targets
            TEXTURE_1D_ARRAY = 0x8C18, // GL_TEXTURE_1D_ARRAY
            PROXY_TEXTURE_1D_ARRAY = 0x8C19, // GL_PROXY_TEXTURE_1D_ARRAY

            // Rectangle texture targets
            TEXTURE_RECTANGLE = 0x84F5, // GL_TEXTURE_RECTANGLE
            PROXY_TEXTURE_RECTANGLE = 0x84F7, // GL_PROXY_TEXTURE_RECTANGLE

            // Cube-map face targets
            TEXTURE_CUBE_MAP_POSITIVE_X = 0x8515, // GL_TEXTURE_CUBE_MAP_POSITIVE_X
            TEXTURE_CUBE_MAP_NEGATIVE_X = 0x8516, // GL_TEXTURE_CUBE_MAP_NEGATIVE_X
            TEXTURE_CUBE_MAP_POSITIVE_Y = 0x8517, // GL_TEXTURE_CUBE_MAP_POSITIVE_Y
            TEXTURE_CUBE_MAP_NEGATIVE_Y = 0x8518, // GL_TEXTURE_CUBE_MAP_NEGATIVE_Y
            TEXTURE_CUBE_MAP_POSITIVE_Z = 0x8519, // GL_TEXTURE_CUBE_MAP_POSITIVE_Z
            TEXTURE_CUBE_MAP_NEGATIVE_Z = 0x851A, // GL_TEXTURE_CUBE_MAP_NEGATIVE_Z

            // Proxy cube-map target
            PROXY_TEXTURE_CUBE_MAP = 0x851B // GL_PROXY_TEXTURE_CUBE_MAP
        };

        void set_data(const void *data, unsigned int width, unsigned int height);
    };

    // 3D texture specialization
    struct Texture3D : public Texture {
        enum Targets3D {
            // 3D texture target
            TEXTURE_3D = 0x806F, // GL_TEXTURE_3D

            // Proxy 3D texture target
            PROXY_TEXTURE_3D = 0x8070, // GL_PROXY_TEXTURE_3D

            // 2D array texture target
            TEXTURE_2D_ARRAY = 0x8C1A, // GL_TEXTURE_2D_ARRAY

            // Proxy 2D array texture target
            PROXY_TEXTURE_2D_ARRAY = 0x8C1B // GL_PROXY_TEXTURE_2D_ARRAY
        };

        void set_data(const void *data, unsigned int width, unsigned int height, unsigned int depth);
    };

    struct SamplerDescriptor {
        enum class Filter {
            Nearest = 0x2600, // GL_NEAREST
            Linear = 0x2601, // GL_LINEAR
        };

        enum class Wrap {
            Repeat = 0x2901, // GL_REPEAT
            ClampToEdge = 0x812F, // GL_CLAMP_TO_EDGE
            MirroredRepeat = 0x8370, // GL_MIRRORED_REPEAT
            ClampToBorder = 0x812D, // GL_CLAMP_TO_BORDER
        };

        Filter min_filter = Filter::Linear;
        Filter mag_filter = Filter::Linear;
        Wrap wrap_s = Wrap::Repeat;
        Wrap wrap_t = Wrap::Repeat;

        bool operator==(const SamplerDescriptor &other) const {
            return min_filter == other.min_filter &&
                   mag_filter == other.mag_filter &&
                   wrap_s == other.wrap_s &&
                   wrap_t == other.wrap_t;
        }
    };
}

#endif //ENGINE24_TEXTURE_H
