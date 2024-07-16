#include <vector>

#include "entt/entt.hpp"
#include "Engine.h"
#include "Graphics.h"
#include "Plugins.h"

int main() {
    Bcg::Engine engine;
    if (!Bcg::Graphics::init()) {
        return -1;
    }

    Bcg::Graphics::set_window_title("TestBed");

    Bcg::Plugins::init();
    Bcg::Plugins::activate_all();
    Bcg::Engine::ExecuteCmdBuffer();

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

    Bcg::Plugins::deactivate_all();

    return 0;
}