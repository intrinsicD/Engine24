//
// Created by alex on 24.07.24.
//

#include "Application.h"
#include "entt/entt.hpp"
#include "Graphics.h"
#include "Plugins.h"

namespace Bcg{
    Application::Application(){
        if (Bcg::Graphics::init()) {

        }
    }

    void Application::init(int width, int height, const char *title){
        Bcg::Graphics::set_window_title(title);
        Bcg::Graphics::set_window_size(width, height);
        Bcg::Plugins::init();
        Bcg::Plugins::activate_all();
        Bcg::Engine::ExecuteCmdBuffer();
    }

    void Application::run(){
        // Game loop
        while (!Bcg::Graphics::should_close()) {
            {
                Bcg::Graphics::poll_events();
                Bcg::Plugins::begin_frame_all();
                Bcg::Plugins::update_all();
                Bcg::Engine::ExecuteCmdBuffer();
            }
            {
                Bcg::Graphics::clear_framebuffer();
                Bcg::Plugins::render_all();
                Bcg::Engine::ExecuteCmdBuffer();
                Bcg::Graphics::start_gui();
                Bcg::Plugins::render_menu();
                Bcg::Plugins::render_gui();
                Bcg::Graphics::render_menu();
                Bcg::Graphics::render_gui();
                Bcg::Graphics::end_gui();
                Bcg::Engine::ExecuteCmdBuffer();
                Bcg::Plugins::end_frame();
                Bcg::Graphics::swap_buffers();
            }
        }
    }

    void Application::cleanup(){
        Bcg::Plugins::deactivate_all();
    }
}