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
#include "ModuleMeshView.h"
#include "ModuleSphereView.h"
#include "GuiModuleCamera.h"
#include "ModulePhongSplattingView.h"
#include "ModuleGraphView.h"
#include "ModuleGraph.h"
#include "ModulePointCloud.h"
#include "TimeAccumulator.h"
#include "Timer.h"
#include "FileWatcher.h"
#include "GuiModuleRendererSettings.h"
#include "TransformSystem.h"
#include "AABBSystem.h"
#include "../../include/Core/Transform/GuiModuleTransforms.h"
#include "GuiModuleHierarchy.h"
#include "GuiModuleMeshLaplacian.h"
#include "GuiModuleGraphLaplacian.h"
#include "GuiModuleGaussianMixture.h"
#include "AssetManager.h"


namespace Bcg {
    Application::Application() {
        platform = std::make_unique<Platform>();
        Engine::Context().emplace<InputManager>();
        Engine::Context().emplace<FileWatcher>();
        Engine::Context().emplace<Application *>(this);
        Engine::Context().emplace<AssetManager>();
    }

    Application::~Application() {
        renderer.reset();
        window.reset();
        platform.reset();
    }

    void Application::init(int width, int height, const char *title) {
        window = std::make_unique<Window>(width, height, title, Engine::Context().get<InputManager>());
        //load renderer & imgui
        auto &asset_manager = Engine::Context().get<AssetManager>();
        renderer = std::make_unique<Renderer>(*window, asset_manager);
        auto viewport_size = renderer->get_viewport().get_size();
        picker_system = std::make_unique<PickerSystem>(width, height);
        auto &entity_selection = engine.state.ctx().emplace<EntitySelection>();
        if (window->exists()) {
            auto &modules = Engine::Context().emplace<Modules>();
            //modules.add(std::make_unique<ModuleGraphics>());
            modules.add(std::make_unique<ModuleMesh>());
            modules.add(std::make_unique<ModuleGraph>());
            modules.add(std::make_unique<ModulePointCloud>());
            modules.add(std::make_unique<ModuleAABB>());
            modules.add(std::make_unique<ModuleCamera>());
            //Rendering Modules
            modules.add(std::make_unique<ModuleMeshView>());
            modules.add(std::make_unique<ModuleSphereView>());
            modules.add(std::make_unique<ModulePhongSplattingView>());
            modules.add(std::make_unique<ModuleGraphView>());

            auto &gui_modules = Engine::Context().emplace<GuiModules>();
            gui_modules.add(std::make_unique<GuiModuleCamera>());
            gui_modules.add(std::make_unique<GuiModuleRendererSettings>(*renderer));
            gui_modules.add(std::make_unique<GuiModuleTransforms>(engine.state, *renderer));
            gui_modules.add(std::make_unique<GuiModuleHierarchy>(engine.state));
            gui_modules.add(std::make_unique<GuiModuleMeshLaplacian>(engine.state));
            gui_modules.add(std::make_unique<GuiModuleGraphLaplacian>(engine.state));
            gui_modules.add(std::make_unique<GuiModuleGaussianMixture>(engine.state));

            Plugins::init();
            Plugins::activate_all();
            Engine::handle_command_double_buffer();
        }
    }

    void Application::run() {
        auto &modules = Engine::Context().get<Modules>();
        modules.activate();

        auto &camera = engine.state.ctx().get<Camera>();

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


            int updates = 0;
            while (accumulator.has_step(k_fixed_time_step) && updates < k_max_updates_per_frame) {
                UpdateTransformSystem(engine.state);
                UpdateWorldAABBSystem(engine.state);
                modules.fixed_update(k_fixed_time_step);
                accumulator.consume_step(k_fixed_time_step);
                ++updates;
            }
            Engine::handle_command_double_buffer();

            double alpha = accumulator.get_alpha(k_fixed_time_step);
            //Second UpdateTransformSystem call to ensure that the transforms are up to date
            UpdateTransformSystem(engine.state);
            UpdateWorldAABBSystem(engine.state);
            modules.begin_frame();
            modules.variable_update(delta_time, alpha);
            modules.update();
            Plugins::begin_frame_all();
            Plugins::update_all();

            Engine::handle_command_double_buffer();
            Engine::handle_buffered_events(); {
                renderer->begin_frame();

                Plugins::render_all();
                modules.render();
                Engine::handle_command_double_buffer();
                Engine::handle_buffered_events();
                renderer->begin_gui();
                GuiModuleTransforms::render_guizmo(*renderer);

                Plugins::render_menu();
                modules.render_menu();
                gui_modules.render_menu();
                Plugins::render_gui();
                modules.render_gui();
                gui_modules.render_gui();
                renderer->end_gui();

                // Apply any transform changes produced by the gizmo/UI this frame
                UpdateTransformSystem(engine.state);
                UpdateWorldAABBSystem(engine.state);

                Engine::handle_command_double_buffer();
                Engine::handle_buffered_events();
                modules.end_frame();
                Plugins::end_frame();
                renderer->end_frame();

                picker_system->update(engine.state, camera);
            }

            ClearWorldAABBDirtyTags(engine.state);
            ClearTransformDirtyTags(engine.state);
        }
    }

    void Application::cleanup() {
        auto &modules = Engine::Context().get<Modules>();
        modules.deactivate();
        auto &gui_modules = Engine::Context().emplace<GuiModules>();
        gui_modules.deactivate();
        Plugins::deactivate_all();
    }
}
