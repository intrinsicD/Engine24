//
// Created by alex on 02.08.24.
//

#include "SphereViewGui.h"
#include "Engine.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "GuiUtils.h"
#include "PropertiesGui.h"
#include "SphereViewCommands.h"

namespace Bcg::Gui {
    void Show(SphereView &view) {
        ImGui::Text("num_spheres: %d", view.num_spheres);
    }

    void ShowSphereView(entt::entity entity_id) {
        if (Engine::valid(entity_id) && Engine::has<SphereView>(entity_id)) {
            auto &view = Engine::State().get<SphereView>(entity_id);
            auto *vertices = GetPrimitives(entity_id).vertices();
            if (vertices) {
                auto properties_3d = vertices->properties(3);
                static std::pair<int, std::string> curr_pos = {0, view.position.bound_buffer_name};
                if (Combo(view.position.shader_name.c_str(), curr_pos, properties_3d)) {
                    Commands::View::SetPositionSphereView(entity_id, properties_3d[curr_pos.first]).execute();
                }

                view.vao.bind();
                {
                    bool enabled_olor = view.color.is_enabled();
                    properties_3d.emplace_back("base_color");
                    static std::pair<int, std::string> curr_color = {0, view.color.bound_buffer_name};

                    if (view.color.bound_buffer_name.empty()) {
                        curr_color.first = properties_3d.size() - 1;
                    }

                    if (!properties_3d.empty() && Combo(view.color.shader_name.c_str(), curr_color, properties_3d)) {
                        if (curr_color.second != "base_color") {
                            Commands::View::SetColorSphereView(entity_id, properties_3d[curr_color.first]).execute();
                        }
                        if (!enabled_olor) {
                            view.color.enable();
                        } else {
                            view.color.disable();
                        }
                    }

                    if (!enabled_olor && (properties_3d.empty() || curr_color.second != "base_color")) {
                        if (ImGui::ColorEdit3("##base_color", view.base_color.data())) {
                            view.color.set_default(view.base_color.data());
                        }
                    }else{
                        ImGui::InputFloat("min_color", &view.min_color);
                        ImGui::InputFloat("max_color", &view.max_color);
                    }
                }

                {
                    bool enabled_radius = view.radius.is_enabled();
                    auto properties_1d = vertices->properties(1);
                    properties_1d.emplace_back("default_radius");
                    static std::pair<int, std::string> curr_radius = {0, view.radius.bound_buffer_name};

                    if (view.radius.bound_buffer_name.empty()) {
                        curr_radius.first = properties_1d.size() - 1;
                    }

                    if (!properties_1d.empty() && Combo(view.radius.shader_name.c_str(), curr_radius, properties_1d)) {
                        if (curr_radius.second != "default_radius") {
                            Commands::View::SetRadiusSphereView(entity_id, properties_1d[curr_radius.first]).execute();
                        }
                        if (!enabled_radius) {
                            view.radius.enable();
                        } else {
                            view.radius.disable();
                        }
                    }

                    if (!enabled_radius && (properties_1d.empty() || curr_radius.second != "default_radius")) {
                        if (ImGui::InputFloat("##default_radius", &view.default_radius)) {
                            view.radius.set_default(&view.default_radius);
                        }
                    }
                }


/*                static std::pair<int, std::string> curr_radius = {0, view.radius.bound_buffer_name};
                auto properties_1d = vertices->properties(1);
                if (Combo(view.radius.shader_name.c_str(), curr_radius, properties_1d)) {
                    Commands::View::SetRadiusSphereView(entity_id, properties_1d[curr_radius.first]).execute();
                }*/

                view.vao.unbind();
            }
            Show(view);
        }
    }
}