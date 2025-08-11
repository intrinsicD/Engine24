//
// Created by alex on 05.08.24.
//


#include "VectorfieldViewGui.h"
#include "Engine.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "GuiUtils.h"
#include "PluginViewVectorfields.h"

namespace Bcg::Gui {
    void Show(VectorfieldView &view) {
        ImGui::Text("num_vectors: %d", view.num_vectors);
    }

    void ShowVectorfieldViews(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) return;
        ImGui::PushID("Views - Vectorfield");
        auto *vertices = GetPrimitives(entity_id).vertices();
        auto properties_3d = vertices->properties({3});
        std::pair<int, std::string> curr = {0, ""};
        if (Combo("create vectorfield", curr, properties_3d)) {
            curr.first = FindIndex(properties_3d, curr.second);
            Commands::Setup<VectorfieldView>(entity_id, properties_3d[curr.first]).execute();
        }
        if (Engine::valid(entity_id) && Engine::has<VectorfieldViews>(entity_id)) {
            auto &views = Engine::State().get<VectorfieldViews>(entity_id);
            ImGui::Checkbox("hide", &views.hide);
            for (auto &item: views.vectorfields) {
                ImGui::PushID(item.first.c_str());
                if (ImGui::CollapsingHeader(item.first.c_str())) {
                    ShowVectorfieldView(entity_id, item.second);
                }
                ImGui::PopID();
            }
        }
        ImGui::PopID();
    }

    void ShowVectorfieldView(entt::entity entity_id, VectorfieldView &view) {
        if (Engine::valid(entity_id)) {
            ImGui::PushID((view.vectorfield_name + view.vector.bound_buffer_name).c_str());
            auto *vertices = GetPrimitives(entity_id).vertices();
            ImGui::Checkbox("hide", &view.hide);
            if (vertices) {
                auto properties_3d = vertices->properties({3});
                static std::pair<int, std::string> curr_pos = {-1, view.position.bound_buffer_name};
                if (curr_pos.first == -1) {
                    curr_pos.first = FindIndex(properties_3d, view.position.bound_buffer_name);
                    if (curr_pos.first == -1) {
                        curr_pos.first = 0;
                    }
                }
                if (Combo(view.position.shader_name.c_str(), curr_pos, properties_3d)) {
                    Commands::SetPositionVectorfieldView(entity_id, view.vectorfield_name,
                                                         properties_3d[curr_pos.first]).execute();
                }

                static std::pair<int, std::string> curr_vector = {-1, view.vector.bound_buffer_name};
                if (curr_vector.first == -1) {
                    curr_vector.first = FindIndex(properties_3d, view.vector.bound_buffer_name);
                    if (curr_vector.first == -1) {
                        curr_vector.first = 0;
                    }
                }
                if (Combo(view.vector.shader_name.c_str(), curr_vector, properties_3d)) {
                    Commands::SetVectorVectorfieldView(entity_id, view.vectorfield_name,
                                                       properties_3d[curr_vector.first]).execute();
                }

                {
                    properties_3d.emplace_back("uniform_color");
                    static std::pair<int, std::string> curr_color = {-1, view.color.bound_buffer_name};
                    if (curr_color.first == -1) {
                        curr_color.first = FindIndex(properties_3d, view.color.bound_buffer_name);
                        if (curr_color.first == -1) {
                            curr_color.first = 0;
                        }
                    }

                    if (Combo(view.color.shader_name.c_str(), curr_color, properties_3d)) {
                        Commands::SetColorVectorfieldView(entity_id, view.vectorfield_name,
                                                          properties_3d[curr_color.first]).execute();
                    }

                    if (view.use_uniform_color) {
                        ImGui::ColorEdit3("##uniform_color_vectorfield_view", glm::value_ptr(view.uniform_color));
                    } else {
                        ImGui::InputFloat("min_color", &view.min_color);
                        ImGui::InputFloat("max_color", &view.max_color);
                    }
                }

                {

                    auto properties_1d = vertices->properties({1});
                    properties_1d.emplace_back("uniform_length");
                    static std::pair<int, std::string> curr_lengths = {-1, view.length.bound_buffer_name};
                    if (curr_lengths.first == -1) {
                        curr_lengths.first = FindIndex(properties_3d, view.length.bound_buffer_name);
                        if (curr_lengths.first == -1) {
                            curr_lengths.first = 0;
                        }
                    }

                    if (Combo(view.length.shader_name.c_str(), curr_lengths, properties_1d)) {
                        Commands::SetLengthVectorfieldView(entity_id, view.vectorfield_name,
                                                           properties_1d[curr_lengths.first]).execute();
                    }

                    if (view.use_uniform_length) {
                        ImGui::InputFloat("##uniform_length", &view.uniform_length);
                    }
                }
            }
            Show(view);
            ImGui::PopID();
        }
    }
}