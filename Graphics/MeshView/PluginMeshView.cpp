//
// Created by alex on 04.08.24.
//


#include "PluginMeshView.h"
#include "Engine.h"
#include "imgui.h"
#include "MeshViewGui.h"
#include "Picker.h"
#include "Camera.h"
#include "Graphics.h"
#include "Transform.h"

namespace Bcg {

    void PluginMeshView::activate() {
        Plugin::activate();
    }

    void PluginMeshView::begin_frame() {}

    void PluginMeshView::update() {}

    void PluginMeshView::end_frame() {}

    void PluginMeshView::deactivate() {
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginMeshView::render_menu() {
        if (ImGui::BeginMenu("Entity")) {
            ImGui::MenuItem("MeshView", nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginMeshView::render_gui() {
        if (show_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("MeshView", &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::ShowMeshView(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void PluginMeshView::render() {
        auto rendergroup = Engine::State().view<MeshView>();
        auto &camera = Engine::Context().get<Camera>();
        for (auto entity_id: rendergroup) {
            auto &view = Engine::State().get<MeshView>(entity_id);

            view.vao.bind();
            view.program.use();
            view.program.set_uniform3fv("light_position", camera.v_params.eye.data());
            view.program.set_uniform1f("min_color", view.min_color);
            view.program.set_uniform1f("max_color", view.max_color);

            if (Engine::has<Transform>(entity_id)) {
                auto &transform = Engine::State().get<Transform>(entity_id);
                view.program.set_uniform4fm("model", transform.data(), false);
            } else {
                view.program.set_uniform4fm("model", Transform().data(), false);
            }

            view.draw();
        }
    }
}