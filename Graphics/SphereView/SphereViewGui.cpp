//
// Created by alex on 02.08.24.
//

#include "SphereViewGui.h"
#include "Engine.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "GuiUtils.h"
#include "SphereViewCommands.h"

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

                static std::pair<int, std::string> curr_normal = {-1, view.normal.bound_buffer_name};
                if(curr_normal.first == -1){
                    curr_normal.first = FindIndex(properties_3d, view.normal.bound_buffer_name);
                    if(curr_normal.first == -1){
                        curr_normal.first = 0;
                    }
                }
                if (Combo(view.normal.shader_name.c_str(), curr_normal, properties_3d)) {
                    Commands::View::SetNormalSphereView(entity_id, properties_3d[curr_normal.first]).execute();
                }

                {
                    properties_3d.emplace_back("uniform_color");
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

                    if (view.use_uniform_color) {
                        ImGui::ColorEdit3("##uniform_color_sphere_view", view.uniform_color.data());
                    } else {
                        ImGui::InputFloat("min_color", &view.min_color);
                        ImGui::InputFloat("max_color", &view.max_color);
                    }
                }

                {
                    auto properties_1d = vertices->properties(1);
                    properties_1d.emplace_back("uniform_radius");
                    static std::pair<int, std::string> curr_radius = {-1, view.radius.bound_buffer_name};
                    if(curr_radius.first == -1){
                        curr_radius.first = FindIndex(properties_1d, view.radius.bound_buffer_name);
                        if(curr_radius.first == -1){
                            curr_radius.first = 0;
                        }
                    }

                    if (Combo(view.radius.shader_name.c_str(), curr_radius, properties_1d)) {
                        Commands::View::SetRadiusSphereView(entity_id, properties_1d[curr_radius.first]).execute();
                    }

                    if (view.use_uniform_radius) {
                        ImGui::InputFloat("##uniform_radius", &view.uniform_radius);
                    }
                }
            }
            Show(view);
            ImGui::PopID();
        }
    }
}