//
// Created by alex on 15.07.24.
//

#include "PluginTransform.h"
#include "imgui.h"
#include "Engine.h"
#include "EventsGui.h"
#include "Picker.h"
#include "RigidTransformGui.h"
#include "Hierarchy.h"

namespace Bcg {
    PluginTransform::PluginTransform() : Plugin("Transform") {}

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
                    if (Engine::State().all_of<Hierarchy>(entity_id)) {
                        auto &hierarchy = Engine::State().get<Hierarchy>(entity_id);
                        if (Engine::valid(hierarchy.parent) && Engine::State().all_of<Transform>(hierarchy.parent)) {
                            auto &parent_transform = Engine::State().get<Transform>(hierarchy.parent);
                            transform.update(parent_transform.matrix());
                        }else{
                            transform.update(Matrix<float, 4, 4>::Identity());
                        }
                    }else{
                        transform.update(Matrix<float, 4, 4>::Identity());
                    }
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