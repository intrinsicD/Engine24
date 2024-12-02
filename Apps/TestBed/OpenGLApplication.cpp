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
#include "MainLoop.h"

namespace Bcg {
    Application::Application() {

    }

    void Application::init(int width, int height, const char *title) {
        Engine::Dispatcher().trigger<Bcg::Events::Initialize>();
        Engine::Context().get<Bcg::Commands::InitializationCommands>().handle();

        if (Bcg::PluginGraphics::init(width, height, title)) {
            Bcg::PluginGraphics::set_window_title(title);
            auto &modules = Engine::Context().emplace<Modules>();
            modules.add(std::make_unique<MeshModule>());

            auto &gui_modules = Engine::Context().emplace<GuiModules>();
            gui_modules.add(std::make_unique<MeshGuiModule>());

            Bcg::Plugins::init();
            Bcg::Plugins::activate_all();
            Bcg::Engine::handle_command_double_buffer();
        }
    }

    void Application::run() {
        Engine::Dispatcher().trigger<Bcg::Events::Startup>();
        Engine::Context().get<Bcg::Commands::StartupCommands>().handle();

        auto &modules = Engine::Context().get<Modules>();
        modules.activate();

        auto &gui_modules = Engine::Context().get<GuiModules>();
        gui_modules.activate();
        // Game loop

        auto &main_loop = Engine::Context().get<Bcg::Commands::MainLoop>();
        while (!Bcg::PluginGraphics::should_close()) {
            main_loop.handle(Engine::Dispatcher());
            {
                Bcg::PluginGraphics::poll_events();
                Bcg::Plugins::begin_frame_all();
                Bcg::Plugins::update_all();
                Bcg::Engine::handle_command_double_buffer();
                Bcg::Engine::handle_buffered_events();
            }
            {
                Bcg::PluginGraphics::clear_framebuffer();
                Bcg::Plugins::render_all();
                Bcg::Engine::handle_command_double_buffer();
                Bcg::Engine::handle_buffered_events();
                Bcg::PluginGraphics::start_gui();
                Bcg::Plugins::render_menu();
                Bcg::Plugins::render_gui();
                gui_modules.render_menu();
                gui_modules.render_gui();
                Bcg::PluginGraphics::end_gui();
                Bcg::Engine::handle_command_double_buffer();
                Bcg::Engine::handle_buffered_events();
                Bcg::Plugins::end_frame();
                Bcg::PluginGraphics::swap_buffers();
            }
        }
    }

    void Application::cleanup() {
        Engine::Dispatcher().trigger<Bcg::Events::Shutdown>();
        Engine::Context().get<Bcg::Commands::ShutdownCommands>().handle();
        auto &modules = Engine::Context().get<Modules>();
        modules.deactivate();
        auto &gui_modules = Engine::Context().emplace<GuiModules>();
        gui_modules.deactivate();
        Bcg::Plugins::deactivate_all();
    }
}