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
        Engine::Dispatcher().trigger<Events::Initialize>();
        Engine::Context().get<Commands::InitializationCommands>().handle();

        if (PluginGraphics::init(width, height, title)) {
            PluginGraphics::set_window_title(title);
            auto &modules = Engine::Context().emplace<Modules>();
            modules.add(std::make_unique<MeshModule>());

            auto &gui_modules = Engine::Context().emplace<GuiModules>();
            gui_modules.add(std::make_unique<MeshGuiModule>());

            Plugins::init();
            Plugins::activate_all();
            Engine::handle_command_double_buffer();
        }
    }

    void Application::run() {
        Engine::Dispatcher().trigger<Events::Startup>();
        Engine::Context().get<Commands::StartupCommands>().handle();

        auto &modules = Engine::Context().get<Modules>();
        modules.activate();

        auto &gui_modules = Engine::Context().get<GuiModules>();
        gui_modules.activate();
        // Game loop

        auto &main_loop = Engine::Context().get<Commands::MainLoop>();
        while (!PluginGraphics::should_close()) {
            main_loop.handle(Engine::Dispatcher());
            {
                PluginGraphics::poll_events();
                Plugins::begin_frame_all();
                Plugins::update_all();
                Engine::handle_command_double_buffer();
                Engine::handle_buffered_events();
            }
            {
                PluginGraphics::clear_framebuffer();
                Plugins::render_all();
                Engine::handle_command_double_buffer();
                Engine::handle_buffered_events();
                PluginGraphics::start_gui();
                Plugins::render_menu();
                Plugins::render_gui();
                gui_modules.render_menu();
                gui_modules.render_gui();
                PluginGraphics::end_gui();
                Engine::handle_command_double_buffer();
                Engine::handle_buffered_events();
                Plugins::end_frame();
                PluginGraphics::swap_buffers();
            }
        }
    }

    void Application::cleanup() {
        Engine::Dispatcher().trigger<Events::Shutdown>();
        Engine::Context().get<Commands::ShutdownCommands>().handle();
        auto &modules = Engine::Context().get<Modules>();
        modules.deactivate();
        auto &gui_modules = Engine::Context().emplace<GuiModules>();
        gui_modules.deactivate();
        Plugins::deactivate_all();
    }
}