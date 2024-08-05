//
// Created by alex on 02.08.24.
//

#include "SphereViewGui.h"
#include "Engine.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "GuiUtils.h"
#include "SphereViewCommands.h"
#include "GLUtils.h"

namespace Bcg::Gui {
    void Show(SphereView &view) {
        ImGui::Text("num_spheres: %d", view.num_spheres);
    }

    void ShowSphereView(entt::entity entity_id) {
        if (Engine::valid(entity_id) && Engine::has<SphereView>(entity_id)) {
            ImGui::PushID("sphere_view");
            auto &view = Engine::State().get<SphereView>(entity_id);
            auto *vertices = GetPrimitives(entity_id).vertices();
            ImGui::Checkbox("hide", &view.hide);
            if (vertices) {
                auto properties_3d = vertices->properties(3);
                static std::pair<int, std::string> curr_pos = {-1, view.position.bound_buffer_name};
                if(curr_pos.first == -1){
                    curr_pos.first = FindIndex(properties_3d, view.position.bound_buffer_name);
                    if(curr_pos.first == -1){
                        curr_pos.first = 0;
                    }
                }
                if (Combo(view.position.shader_name.c_str(), curr_pos, properties_3d)) {
                    Commands::View::SetPositionSphereView(entity_id, properties_3d[curr_pos.first]).execute();
                }

                {
                    properties_3d.emplace_back("base_color");
                    static std::pair<int, std::string> curr_color = {-1, view.color.bound_buffer_name};
                    if(curr_color.first == -1){
                        curr_color.first = FindIndex(properties_3d, view.color.bound_buffer_name);
                        if(curr_color.first == -1){
                            curr_color.first = 0;
                        }
                    }

                    if (Combo(view.color.shader_name.c_str(), curr_color, properties_3d)) {
                        Commands::View::SetColorSphereView(entity_id, properties_3d[curr_color.first]).execute();
                    }

                    view.vao.bind();
                    bool enabled_color = view.color.is_enabled();
                    view.vao.unbind();

                    if (!enabled_color) {
                        if (ImGui::ColorEdit3("##base_color_sphere_view", view.base_color.data())) {
                            view.vao.bind();
                            view.color.set_default(view.base_color.data());
                            view.color.disable();
                            view.vao.unbind();
                        }
                    } else {
                        ImGui::InputFloat("min_color", &view.min_color);
                        ImGui::InputFloat("max_color", &view.max_color);
                    }
                }

                {

                    auto properties_1d = vertices->properties(1);
                    properties_1d.emplace_back("default_radius");
                    static std::pair<int, std::string> curr_radius = {-1, view.radius.bound_buffer_name};
                    if(curr_radius.first == -1){
                        curr_radius.first = FindIndex(properties_3d, view.radius.bound_buffer_name);
                        if(curr_radius.first == -1){
                            curr_radius.first = 0;
                        }
                    }

                    if (Combo(view.radius.shader_name.c_str(), curr_radius, properties_1d)) {
                        Commands::View::SetRadiusSphereView(entity_id, properties_1d[curr_radius.first]).execute();
                    }

                    view.vao.bind();
                    bool enabled_radius = view.radius.is_enabled();
                    view.vao.unbind();

                    if (!enabled_radius) {
                        if (ImGui::InputFloat("##default_radius", &view.default_radius)) {
                            view.vao.bind();
                            view.radius.set_default(&view.default_radius);
                            view.radius.disable();
                            view.vao.unbind();
                        }
                    }
                }
            }
            Show(view);
            ImGui::PopID();
        }
    }
}