//
// Created by alex on 15.07.24.
//

#include "PluginTransform.h"
#include "imgui.h"
#include "Engine.h"
#include "EventsGui.h"
#include "Picker.h"
#include "RigidTransformGui.h"
#include "PluginHierarchy.h"
#include "CommandDoubleBuffer.h"
#include "PluginHierarchy.h"

namespace Bcg {

    PluginTransform::PluginTransform() : Plugin("Transform") {}

    Transform *PluginTransform::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return nullptr; }
        if (Engine::has<Transform>(entity_id)) { return &Engine::State().get<Transform>(entity_id); }

        Log::Info("Transform setup for entity: {}", entity_id);
        return &Engine::State().emplace<Transform>(entity_id, Transform());
    }

    void PluginTransform::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return; }
        if (!Engine::has<Transform>(entity_id)) { return; }

        Engine::State().remove<Transform>(entity_id);
        Log::Info("Transform cleanup for entity: {}", entity_id);
    }

    void PluginTransform::activate() {
        Plugin::activate();
    }

    void PluginTransform::begin_frame() {

    }

    void PluginTransform::update() {

    }

    void PluginTransform::end_frame() {

    }

    void PluginTransform::deactivate() {
        Plugin::deactivate();
    }

    static bool show_gui = false;

    static void on_gui_render(const Events::Gui::Render &event) {
        if (!show_gui) {
            Engine::Dispatcher().sink<Events::Gui::Render>().disconnect<on_gui_render>();
            return;
        }

        auto &picked = Engine::Context().get<Picked>();
        auto entity_id = picked.entity.id;
        if (ImGui::Begin("Transform", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
            if (Engine::valid(entity_id) && Engine::State().all_of<Transform>(entity_id)) {
                auto &transform = Engine::State().get<Transform>(entity_id);
                if (Gui::Show(transform)) {
                    PluginHierarchy::mark_transforms_dirty(entity_id);
                    auto &double_cmd_buffer = Engine::Context().get<DoubleCommandBuffer>();
                    double_cmd_buffer.current().add_command(Commands::UpdateTransformsDeferred(entity_id));
                }
            }
        }
        ImGui::End();
    }

    void PluginTransform::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            if (ImGui::MenuItem(name, nullptr, &show_gui)) {
                Engine::Dispatcher().sink<Events::Gui::Render>().connect<on_gui_render>();
            }
            ImGui::EndMenu();
        }
    }

    void PluginTransform::render_gui() {


    }

    void PluginTransform::render() {

    }
}