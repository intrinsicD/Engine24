//
// Created by alex on 08.11.24.
//

#include "KDTreeGui.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "PluginViewSphere.h"
#include "Engine.h"

namespace Bcg::Gui {
    void ShowKDTree(entt::entity entity_id) {
        cuda::KDTreeCuda kdtree(entity_id);
        if (kdtree) {
            static int max_level = -1;
            if (ImGui::Button("Build Samples")) {
                kdtree.fill_samples();
                max_level = kdtree.compute_num_levels();
            }
            static int level = 0;
            if (max_level > -1) {
                ImGui::Text("Max Level: %d", max_level);
            }
            if (ImGui::InputInt("Level", &level)) {
                level = std::max(0, std::min(level, max_level));
            }
            if (ImGui::Button("Sample level")) {
                auto samples = kdtree.get_samples(level);

                auto vertices = GetPrimitives(entity_id).vertices();
                auto v_color = vertices->get_or_add<Eigen::Vector<float, 3>>("v_samples", Eigen::Vector<float, 3>::Zero());
                std::fill(v_color.vector().begin(), v_color.vector().end(), Eigen::Vector<float, 3>::Zero());
                auto v_radius = vertices->get_or_add<float>("v_radius", 0.0f);

                auto &view = Engine::State().get<SphereView>(entity_id);


                std::fill(v_radius.vector().begin(), v_radius.vector().end(), view.uniform_radius);

                for (size_t i = 0; i < samples.size(); ++i) {
                    v_color[samples[i]] = Eigen::Vector<float, 3>(1.0f, 0.0f, 0.0f);
                    v_radius[samples[i]] = view.uniform_radius + 5;
                }
                Commands::SetColorSphereView cmd(entity_id, "v_samples");
                Commands::SetRadiusSphereView cmd2(entity_id, "v_radius");
                cmd.execute();
                cmd2.execute();
            }
        }
    }

    void Show(cuda::KDTreeCuda &kdtree) {

    }
}