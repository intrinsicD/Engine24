//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_PLUGINGRAPHICS_H
#define ENGINE24_PLUGINGRAPHICS_H

#include "MatVec.h"
#include "entt/fwd.hpp"
#include "Plugin.h"
#include "Command.h"

struct GLFWwindow;

namespace Bcg {

    class PluginGraphics : public Plugin {
    public:
        PluginGraphics();

        ~PluginGraphics() override = default;

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;

        static bool init(int width, int height, const char *title);

        static bool should_close();

        static void poll_events();

        static void set_window_title(const char *title);

        static void set_clear_color(const float *color);

        static void clear_framebuffer();

        static void start_gui();

        static void end_gui();

        static void swap_buffers();

        static Eigen::Vector<int, 2> get_window_pos();

        static Eigen::Vector<int, 2> get_window_size();

        static Eigen::Vector<int, 2> get_framebuffer_size();

        static Eigen::Vector<int, 4> get_viewport();

        static Eigen::Vector<int, 4> get_viewport_dpi_adjusted();

        static float dpi_scaling();

        static bool read_depth_buffer(int x, int y, float &z);

        //--------------------------------------------------------------------------------------------------------------
    };

}

#endif //ENGINE24_PLUGINGRAPHICS_H
