//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_GRAPHICS_H
#define ENGINE24_GRAPHICS_H

#include <unordered_map>
#include <vector>
#include <string>
#include "MatVec.h"

struct GLFWwindow;

namespace Bcg {

    class Graphics {
    public:
        static bool init();

        static bool should_close();

        static void poll_events();

        static void set_window_title(const char *title);

        static void set_clear_color(const float *color);

        static void clear_framebuffer();

        static void start_gui();

        static void render_menu();

        static void render_gui();

        static void end_gui();

        static void swap_buffers();

        static Vector<int, 2> get_window_pos();

        static Vector<int, 2> get_window_size();

        static Vector<int, 2> get_framebuffer_size();

        static Vector<int, 4> get_viewport();

        static Vector<int, 4> get_viewport_dpi_adjusted();

        static bool read_depth_buffer(int x, int y, float &z);

        static float dpi_scaling();

        //--------------------------------------------------------------------------------------------------------------
    };
}

#endif //ENGINE24_GRAPHICS_H
