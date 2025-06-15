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
#include "TimeAccumulator.h"
#include "Timer.h"
#include "GLFWContext.h"
#include "FileWatcher.h"

namespace Bcg {
    Application::Application() {
        Engine::Context().emplace<GLFWContext>().init();
        Engine::Context().emplace<InputManager>();
        Engine::Context().emplace<FileWatcher>();
        Engine::Context().emplace<Application *>(this);
    }

    void Application::init(int width, int height, const char *title) {
        window = std::make_unique<Window>(width, height, title, Engine::Context().get<InputManager>());
        //load renderer & imgui
        renderer = std::make_unique<Renderer>(*window);
        if (window->exists()) {
            auto &modules = Engine::Context().emplace<Modules>();
            //modules.add(std::make_unique<ModuleGraphics>());
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

            Plugins::init();
            Plugins::activate_all();
            Engine::handle_command_double_buffer();
        }
        /*if (ModuleGraphics::init(width, height, title)) {

        }*/
    }

    void Application::run() {
        auto &modules = Engine::Context().get<Modules>();
        modules.activate();

        auto &gui_modules = Engine::Context().get<GuiModules>();
        gui_modules.activate();
        // Game loop

        TimeAccumulator accumulator;
        const double k_fixed_time_step = 1.0 / 60.0;
        const int k_max_updates_per_frame = 5;

        TimeTicker main_loop_ticker;
        while (!window->should_close()) {
            double delta_time = main_loop_ticker.tick();
            accumulator.add(delta_time);
            window->poll_events();
            /*ModuleGraphics::poll_events();*/

            int updates = 0;
            while (accumulator.has_step(k_fixed_time_step) && updates < k_max_updates_per_frame) {
                modules.fixed_update(k_fixed_time_step);
                accumulator.consume_step(k_fixed_time_step);
                ++updates;
            }
            Engine::handle_command_double_buffer();

            double alpha = accumulator.get_alpha(k_fixed_time_step);
            modules.begin_frame();
            modules.variable_update(delta_time, alpha);
            modules.update();
            Plugins::begin_frame_all();
            Plugins::update_all();

            Engine::handle_command_double_buffer();
            Engine::handle_buffered_events();
            {
                renderer->begin_frame();
                /*ModuleGraphics::clear_framebuffer();*/
                Plugins::render_all();
                modules.render();
                Engine::handle_command_double_buffer();
                Engine::handle_buffered_events();
                renderer->begin_gui();
                /*ModuleGraphics::start_gui();*/
                Plugins::render_menu();
                modules.render_menu();
                gui_modules.render_menu();
                Plugins::render_gui();
                modules.render_gui();
                gui_modules.render_gui();
                renderer->end_gui();
                /*ModuleGraphics::end_gui();*/
                Engine::handle_command_double_buffer();
                Engine::handle_buffered_events();
                modules.end_frame();
                Plugins::end_frame();
                renderer->end_frame();
                /*ModuleGraphics::swap_buffers();*/
            }
        }
    }

    void Application::cleanup() {
        auto &modules = Engine::Context().get<Modules>();
        modules.deactivate();
        auto &gui_modules = Engine::Context().emplace<GuiModules>();
        gui_modules.deactivate();
        Plugins::deactivate_all();
        Engine::Context().get<GLFWContext>().shutdown();
    }
}