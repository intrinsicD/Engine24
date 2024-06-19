//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_GRAPHICS_H
#define ENGINE24_GRAPHICS_H

namespace Bcg {
    class Graphics {
    public:
        bool init();

        bool should_close() const;

        void poll_events() const;

        void set_clear_color(const float *color);

        void clear_framebuffer() const;

        void start_gui() const;

        void render_menu() const;

        void render_gui() const;

        void end_gui() const;

        void swap_buffers() const;
    };
}

#endif //ENGINE24_GRAPHICS_H
