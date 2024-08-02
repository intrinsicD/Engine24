//
// Created by alex on 02.08.24.
//

#include "SphereViewGui.h"
#include "Engine.h"
#include "imgui.h"

namespace Bcg::Gui {
    void Show(SphereView &view) {
        if (ImGui::ColorEdit3("base_color", view.base_color.data())) {
            view.vao.bind();
            view.color.disable();
            view.color.set_default(view.base_color.data());
            view.vao.unbind();
        }
        if (!view.color.bound_buffer_name.empty()) {
            ImGui::Text("v_color: %s", view.color.bound_buffer_name.c_str());
            view.vao.bind();
            bool enabled = view.color.is_enabled();
            if (ImGui::Checkbox("Enable color attribute", &enabled)) {
                if (enabled) {
                    view.color.enable();
                } else {
                    view.color.disable();
                }
            }
            view.vao.unbind();
        }

        if (!view.radius.bound_buffer_name.empty()) {
            ImGui::Text("v_radius: %s", view.radius.bound_buffer_name.c_str());
            view.vao.bind();
            bool enabled = view.radius.is_enabled();
            if (ImGui::Checkbox("Enable radius attribute", &enabled)) {
                if (enabled) {
                    view.radius.enable();
                } else {
                    view.radius.disable();
                }
            }
            view.vao.unbind();
        }
        ImGui::Text("num_spheres: %d", view.num_spheres);
        if (ImGui::InputFloat("default_radius", &view.default_radius)) {
            view.vao.bind();
            view.radius.disable();
            view.radius.set_default(&view.default_radius);
            view.vao.unbind();
        }
        ImGui::InputFloat("min_color", &view.min_color);
        ImGui::InputFloat("max_color", &view.max_color);
    }

    void ShowSphereView(entt::entity entity_id) {
        if (Engine::valid(entity_id) && Engine::has<SphereView>(entity_id)) {
            auto &view = Engine::State().get<SphereView>(entity_id);
            Show(view);
        }
    }
}