//
// Created by alex on 04.08.24.
//

#include "MeshViewGui.h"
#include "imgui.h"
#include "Engine.h"
#include "Picker.h"
#include "GetPrimitives.h"
#include "MeshViewCommands.h"
#include "GLUtils.h"

namespace Bcg::Gui {
    void Show(MeshView &view) {
        ImGui::Text("num_indices: %d", view.num_indices);
    }

    void ShowMeshView(entt::entity entity_id) {
        if (Engine::valid(entity_id) && Engine::has<MeshView>(entity_id)) {
            ImGui::PushID("mesh_view");
            auto &view = Engine::State().get<MeshView>(entity_id);
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
                    Commands::View::SetPositionMeshView(entity_id, properties_3d[curr_pos.first]).execute();
                }

                static std::pair<int, std::string> curr_normal = {-1, view.normal.bound_buffer_name};
                if(curr_normal.first == -1){
                    curr_normal.first = FindIndex(properties_3d, view.normal.bound_buffer_name);
                    if(curr_normal.first == -1){
                        curr_normal.first = 0;
                    }
                }
                if (Combo(view.normal.shader_name.c_str(), curr_normal, properties_3d)) {
                    Commands::View::SetNormalMeshView(entity_id, properties_3d[curr_normal.first]).execute();
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
                        Commands::View::SetColorMeshView(entity_id, properties_3d[curr_color.first]).execute();
                    }

                    view.vao.bind();
                    bool enabled_color = view.color.is_enabled();
                    view.vao.unbind();

                    if (!enabled_color) {
                        if (ImGui::ColorEdit3("##base_color_mesh_view", view.base_color.data())) {
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
            }
            Show(view);
            ImGui::PopID();
        }
    }
}
