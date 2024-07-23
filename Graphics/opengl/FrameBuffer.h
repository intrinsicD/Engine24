//
// Created by alex on 22.07.24.
//

#ifndef ENGINE24_FRAMEBUFFER_H
#define ENGINE24_FRAMEBUFFER_H

#include "Texture.h"

namespace Bcg {
    struct FrameBuffer {
        unsigned int id = 0;

        enum Targets {
            FRAMEBUFFER = 0x8D40,
            READ_FRAMEBUFFER = 0x8CA8,
            DRAW_FRAMEBUFFER = 0x8CA8,
        } target = FRAMEBUFFER;

        void create();

        void destroy();

        void bind() const;

        void unbind() const;

        [[nodiscard]] bool check() const;

        void blit(int srcX0, int srcY0, int srcX1, int srcY1,
                  int dstX0, int dstY0, int dstX1, int dstY1,
                  unsigned int mask, unsigned int filter,
                  const FrameBuffer &other) const;

        void read_pixels(int x, int y, int width, int height, unsigned int format, unsigned int type, void *data) const;

        unsigned int get_max_color_attachments() const;

        void add_texture_2d(const Texture2D &texture2D);
    };
}

#endif //ENGINE24_FRAMEBUFFER_H
