//
// Created by alex on 24.07.24.
//

#include "Application.h"
#include "entt/entt.hpp"
#include "ModuleGraphics.h"
#include "Plugins.h"
#include "GuiModules.h"
#include "Modules.h"
#include "ModuleMesh.h"
#include "ModuleAABB.h"
#include "ModuleCamera.h"
#include "ModuleTransform.h"
#include "ModuleMeshView.h"
#include "ModuleSphereView.h"
#include "GuiModuleCamera.h"
#include "ModulePhongSplattingView.h"
#include "ModuleGraphView.h"
#include "MainLoop.h"

namespace Bcg {
    Application::Application() {

    }

    void Application::init(int width, int height, const char *title) {
        Engine::Dispatcher().trigger<Bcg::Events::Initialize>();
        Engine::Context().get<Bcg::Commands::InitializationCommands>().handle();

        if (Bcg::ModuleGraphics::init(width, height, title)) {
            Bcg::ModuleGraphics::set_window_title(title);
            auto &modules = Engine::Context().emplace<Modules>();
            modules.add(std::make_unique<ModuleGraphics>());
            modules.add(std::make_unique<ModuleMesh>());
            modules.add(std::make_unique<ModuleAABB>());
            modules.add(std::make_unique<ModuleCamera>());
            modules.add(std::make_unique<ModuleTransform>());
            //Rendering Modules
            modules.add(std::make_unique<ModuleMeshView>());
            modules.add(std::make_unique<ModuleSphereView>());
            modules.add(std::make_unique<ModulePhongSplattingView>());
            modules.add(std::make_unique<ModuleGraphView>());

            auto &gui_modules = Engine::Context().emplace<GuiModules>();
            gui_modules.add(std::make_unique<GuiModuleCamera>());

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
        while (!Bcg::ModuleGraphics::should_close()) {
            main_loop.handle(Engine::Dispatcher());
            {
                Bcg::ModuleGraphics::poll_events();
                Bcg::Plugins::begin_frame_all();
                modules.begin_frame();
                Bcg::Plugins::update_all();
                modules.update();
                Bcg::Engine::handle_command_double_buffer();
                Bcg::Engine::handle_buffered_events();
            }
            {
                Bcg::ModuleGraphics::clear_framebuffer();
                Bcg::Plugins::render_all();
                modules.render();
                Bcg::Engine::handle_command_double_buffer();
                Bcg::Engine::handle_buffered_events();
                Bcg::ModuleGraphics::start_gui();
                Bcg::Plugins::render_menu();
                modules.render_menu();
                gui_modules.render_menu();
                Bcg::Plugins::render_gui();
                modules.render_gui();
                gui_modules.render_gui();
                Bcg::ModuleGraphics::end_gui();
                Bcg::Engine::handle_command_double_buffer();
                Bcg::Engine::handle_buffered_events();
                Bcg::Plugins::end_frame();
                modules.end_frame();
                Bcg::ModuleGraphics::swap_buffers();
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