//
// Created by alex on 6/16/25.
//

#ifndef RENDERER_H
#define RENDERER_H

#include "Window.h"

namespace Bcg {
    class Renderer {
    public:
        explicit Renderer(Window &window);

        ~Renderer();

        void begin_frame();

        void end_frame();

        void set_clear_color(const Vector<float, 4> &color);

        const Vector<float, 4> &get_clear_color() const;

        void clear_framebuffer();

        void begin_gui();

        void end_gui();

        Vector<float, 4> get_viewport() const;

        Vector<float, 4> get_viewport_dpi_adjusted() const;

    private:
        void init_graphics();

        void init_imgui();

        Window &m_window;
        Vector<float, 4> m_clear_color = {0.2f, 0.3f, 0.3f, 1.0f}; // Default clear color
    };
}
#endif //RENDERER_H
