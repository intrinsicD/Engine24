//
// Created by alex on 22.07.24.
//

#ifndef ENGINE24_TEXTURE_H
#define ENGINE24_TEXTURE_H

namespace Bcg {
    struct Texture {
        unsigned int id;
        unsigned int target;
        int level;
        unsigned int width;
        unsigned int height;
        int border;
        unsigned int format;
        unsigned int type;

        enum class Formats {
            DEPTH_COMPONENT,
            DEPTH_STENCIL_COMPONENT,
            RED,
            RG,
            RGB,
            RGBA,
            BGR,
            BGRA
        };

        enum class InternalFormats {
// Base internal formats
            ALPHA,
            LUMINANCE,
            LUMINANCE_ALPHA,
            INTENSITY,
            RED,
            RG,
            RGB,
            RGBA,

// Sized internal formats
            R8,
            R8_SNORM,
            R16,
            R16_SNORM,
            RG8,
            RG8_SNORM,
            RG16,
            RG16_SNORM,
            R3_G3_B2,
            RGB4,
            RGB5,
            RGB8,
            RGB8_SNORM,
            RGB10,
            RGB12,
            RGB16_SNORM,
            RGBA2,
            RGBA4,
            RGB5_A1,
            RGBA8,
            RGBA8_SNORM,
            RGB10_A2,
            RGB10_A2UI,
            RGBA12,
            RGBA16,
            SRGB8,
            SRGB8_ALPHA8,
            R16F,
            RG16F,
            RGB16F,
            RGBA16F,
            R32F,
            RG32F,
            RGB32F,
            RGBA32F,
            R11F_G11F_B10F,
            RGB9_E5,
            R8I,
            R8UI,
            R16I,
            R16UI,
            R32I,
            R32UI,
            RG8I,
            RG8UI,
            RG16I,
            RG16UI,
            RG32I,
            RG32UI,
            RGB8I,
            RGB8UI,
            RGB16I,
            RGB16UI,
            RGB32I,
            RGB32UI,
            RGBA8I,
            RGBA8UI,
            RGBA16I,
            RGBA16UI,
            RGBA32I,
            RGBA32UI,

// Depth and stencil formats
            DEPTH_COMPONENT,
            DEPTH_STENCIL,
            DEPTH_COMPONENT16,
            DEPTH_COMPONENT24,
            DEPTH_COMPONENT32,
            DEPTH_COMPONENT32F,
            DEPTH24_STENCIL8,
            DEPTH32F_STENCIL8,

// Compressed internal formats
            COMPRESSED_RED,
            COMPRESSED_RG,
            COMPRESSED_RGB,
            COMPRESSED_RGBA,
            COMPRESSED_SRGB,
            COMPRESSED_SRGB_ALPHA,
            COMPRESSED_RED_RGTC1,
            COMPRESSED_SIGNED_RED_RGTC1,
            COMPRESSED_RG_RGTC2,
            COMPRESSED_SIGNED_RG_RGTC2,
            COMPRESSED_RGBA_BPTC_UNORM,
            COMPRESSED_SRGB_ALPHA_BPTC_UNORM,
            COMPRESSED_RGB_BPTC_SIGNED_FLOAT,
            COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT,
        };

        void create();

        void destroy();
    };

    struct Texture1D : public Texture {
        enum Targets {
            TEXTURE_1D,
            PROXY_TEXTURE_1D
        };
    };

    struct Texture2D : public Texture {
        enum Targets {
            TEXTURE_2D,
            PROXY_TEXTURE_2D,
            TEXTURE_1D_ARRAY,
            PROXY_TEXTURE_1D_ARRAY,
            TEXTURE_RECTANGLE,
            PROXY_TEXTURE_RECTANGLE,
            TEXTURE_CUBE_MAP_POSITIVE_X,
            TEXTURE_CUBE_MAP_NEGATIVE_X,
            TEXTURE_CUBE_MAP_POSITIVE_Y,
            TEXTURE_CUBE_MAP_NEGATIVE_Y,
            TEXTURE_CUBE_MAP_POSITIVE_Z,
            TEXTURE_CUBE_MAP_NEGATIVE_Z,
            PROXY_TEXTURE_CUBE_MAP
        };
    };

    struct Texture3D : public Texture {
        enum Targets {
            TEXTURE_3D,
            PROXY_TEXTURE_3D,
            TEXTURE_2D_ARRAY,
            PROXY_TEXTURE_2D_ARRAY,
        };
    };
}

#endif //ENGINE24_TEXTURE_H
