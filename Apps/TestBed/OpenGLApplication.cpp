//
// Created by alex on 24.07.24.
//

#include "Application.h"
#include "entt/entt.hpp"
#include "PluginGraphics.h"
#include "Plugins.h"

namespace Bcg{
    Application::Application(){

    }

    void Application::init(int width, int height, const char *title){
        if (Bcg::PluginGraphics::init(width, height, title)) {
            Bcg::PluginGraphics::set_window_title(title);
            Bcg::Plugins::init();
            Bcg::Plugins::activate_all();
            Bcg::Engine::ExecuteCmdBuffer();
        }
    }

    void Application::run(){
        // Game loop
        while (!Bcg::PluginGraphics::should_close()) {
            {
                Bcg::PluginGraphics::poll_events();
                Bcg::Plugins::begin_frame_all();
                Bcg::Plugins::update_all();
                Bcg::Engine::ExecuteCmdBuffer();
            }
            {
                Bcg::PluginGraphics::clear_framebuffer();
                Bcg::Plugins::render_all();
                Bcg::Engine::ExecuteCmdBuffer();
                Bcg::PluginGraphics::start_gui();
                Bcg::Plugins::render_menu();
                Bcg::Plugins::render_gui();
                Bcg::PluginGraphics::end_gui();
                Bcg::Engine::ExecuteCmdBuffer();
                Bcg::Plugins::end_frame();
                Bcg::PluginGraphics::swap_buffers();
            }
        }
    }

    void Application::cleanup(){
        Bcg::Plugins::deactivate_all();
    }
}