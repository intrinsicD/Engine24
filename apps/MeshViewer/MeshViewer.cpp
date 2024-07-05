//
// Created by alex on 05.07.24.
//

#include "MeshViewer.h"
#include "Engine.h"
#include "Graphics.h"
#include "Plugins.h"

namespace Bcg {
    MeshViewer::MeshViewer() {
        Engine engine;
    }

    void MeshViewer::run() {
        if (!Graphics::init()) {
            return ;
        }

        Plugins::init();
        Plugins::activate_all();
        Engine::ExecuteCmdBuffer();

        // Game loop
        while (!Graphics::should_close()) {
            {
                Graphics::poll_events();
                Plugins::begin_frame_all();
                Plugins::update_all();
                Engine::ExecuteCmdBuffer();
            }
            {
                Graphics::clear_framebuffer();
                Plugins::render_all();
                Engine::ExecuteCmdBuffer();
                Graphics::start_gui();
                Plugins::render_menu();
                Plugins::render_gui();
                Graphics::render_menu();
                Graphics::render_gui();
                Graphics::end_gui();
                Engine::ExecuteCmdBuffer();
                Plugins::end_frame();
                Graphics::swap_buffers();
            }
        }

        Plugins::deactivate_all();
    }
}