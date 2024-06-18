//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGINGLFWWINDOW_H
#define ENGINE24_PLUGINGLFWWINDOW_H

#include "Plugin.h"

namespace Bcg{
    class PluginGlfwWindow : public Plugin{
    public:
        PluginGlfwWindow() = default;

        ~PluginGlfwWindow() override = default;

        bool init();

        void clear_framebuffer();

        void start_gui();

        void end_gui();

        void activate() override;

        void update() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };
}

#endif //ENGINE24_PLUGINGLFWWINDOW_H
