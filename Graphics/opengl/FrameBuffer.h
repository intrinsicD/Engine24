//
// Created by alex on 22.07.24.
//

#ifndef ENGINE24_FRAMEBUFFER_H
#define ENGINE24_FRAMEBUFFER_H

#include "Texture.h"
#include <vector>

namespace Bcg {
    class Framebuffer {
    public:
        Framebuffer(uint32_t width, uint32_t height);

        ~Framebuffer();

        // Prevent copying to avoid issues with OpenGL handle ownership
        Framebuffer(const Framebuffer &) = delete;

        Framebuffer &operator=(const Framebuffer &) = delete;

        // Allow moving
        Framebuffer(Framebuffer &&other) noexcept;

        Framebuffer &operator=(Framebuffer &&other) noexcept;

        void bind() const;

        void unbind() const;

        void resize(uint32_t width, uint32_t height);

        // Reads the integer value of a single pixel from the framebuffer.
        int read_pixel(uint32_t x, uint32_t y);

        uint32_t get_width() const { return m_width; }

        uint32_t get_height() const { return m_height; }

    private:
        void invalidate(); // Recreate textures and re-attach them

        uint32_t m_renderer_id = 0;
        uint32_t m_color_attachment = 0; // The texture that stores entity IDs
        uint32_t m_depth_attachment = 0; // The depth buffer

        uint32_t m_width;
        uint32_t m_height;
    };
}

#endif //ENGINE24_FRAMEBUFFER_H
