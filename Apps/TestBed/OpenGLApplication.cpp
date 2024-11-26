//
// Created by alex on 24.07.24.
//

#include "Application.h"
#include "entt/entt.hpp"
#include "PluginGraphics.h"
#include "Plugins.h"
#include "GuiModules.h"
#include "Modules.h"
#include "MeshModule.h"
#include "MeshGuiModule.h"

namespace Bcg{
    Application::Application(){

    }

    void Application::init(int width, int height, const char *title){
        if (Bcg::PluginGraphics::init(width, height, title)) {
            Bcg::PluginGraphics::set_window_title(title);
            auto &modules = Engine::Context().emplace<Modules>();
            auto &gui_modules = Engine::Context().emplace<GuiModules>();
            modules.add(std::make_unique<MeshModule>());
            gui_modules.add(std::make_unique<MeshGuiModule>());
            Bcg::Plugins::init();
            Bcg::Plugins::activate_all();
            Bcg::Engine::ExecuteCmdBuffer();
        }
    }

    void Application::run(){
        auto &gui_modules = Engine::Context().get<GuiModules>();
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
                gui_modules.render_menu();
                gui_modules.render_gui();
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