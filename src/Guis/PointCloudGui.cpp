//
// Created by alex on 30.07.24.
//


#include "PointCloudGui.h"
#include "ImGuiFileDialog.h"
#include "PluginPointCloud.h"
#include "PropertiesGui.h"
#include "Engine.h"
#include "PluginPointCloud.h"
#include "GetPrimitives.h"

namespace Bcg::Gui {
    void ShowLoadPointCloud() {
        if (ImGuiFileDialog::Instance()->Display("Load PointCloud", ImGuiWindowFlags_NoCollapse, ImVec2(200, 100))) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                // action
                auto mesh = PluginPointCloud::load(filePathName);
            }

            // close
            ImGuiFileDialog::Instance()->Close();
        }
    }

    void ShowPointCloud(entt::entity entity_id) {
        if (Engine::valid(entity_id)) {
            auto *vertices = GetPrimitives(entity_id).vertices();
            if (vertices) {
                Show("Vertices", *vertices);
                ImGui::Separator();
                static int num_closest = 12;
                ImGui::InputInt("num_closest", &num_closest);
                if (ImGui::Button("LocalPcaKnn")) {
                    Commands::ComputePointCloudLocalPcasKnn(entity_id, num_closest).execute();
                }
                static int k = 12;
                static int iterations = 100;
                ImGui::InputInt("k", &k);
                ImGui::InputInt("iterations", &iterations);
                if (ImGui::Button("Kmeans")) {
                    Commands::ComputeKMeans(entity_id, k, iterations).execute();
                }
                if (ImGui::Button("HierarchicalKmeans")) {
                    Commands::ComputeHierarchicalKMeans(entity_id, k, iterations).execute();
                }
                if (ImGui::Button("LocalGaussians")) {
                    Commands::ComputeLocalGaussians(entity_id, k).execute();
                }
            }
        }
    }

    void Show(PointCloud &pc) {
        if (ImGui::CollapsingHeader(("Vertices #v: " + std::to_string(pc.n_vertices())).c_str())) {
            ImGui::PushID("Vertices");
            Show("##Vertices", pc.vprops_);
            ImGui::PopID();
        }
    }
}