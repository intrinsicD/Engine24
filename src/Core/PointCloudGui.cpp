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
#include "CommandsPointCloud.h"

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
                if (ImGui::CollapsingHeader(("Vertices #v: " + std::to_string(vertices->size())).c_str())) {
                    ImGui::PushID("Vertices");
                    Show("##Vertices", *vertices);
                    ImGui::PopID();
                }
                ImGui::Separator();
                static int num_closest = 12;
                ImGui::InputInt("num_closest", &num_closest);
                if (ImGui::Button("LocalPcaKnn")) {
                    Commands::ComputePointCloudLocalPcasKnn(entity_id, num_closest).execute();
                }
                static int iterations = 100;
                ImGui::InputInt("iterations", &iterations);
                if (ImGui::Button("Kmeans")) {
                    Commands::ComputeKMeans(entity_id, num_closest, iterations).execute();
                }
                if (ImGui::Button("HierarchicalKmeans")) {
                    Commands::ComputeHierarchicalKMeans(entity_id, num_closest, iterations).execute();
                }
                if (ImGui::Button("LocalGaussians")) {
                    Commands::ComputeLocalGaussians(entity_id, num_closest).execute();
                }
                static int levels = 5;
                ImGui::InputInt("levels", &levels);
                if(ImGui::Button("Hem")){
                    Commands::ComputeHem(entity_id, levels, num_closest).execute();
                }
            }
        }
    }

    void Show(PointCloud &pc) {
        if (ImGui::CollapsingHeader(("Vertices #v: " + std::to_string(pc.data.vertices.n_vertices())).c_str())) {
            ImGui::PushID("Vertices");
            Show("##Vertices", pc.data.vertices);
            ImGui::PopID();
        }
    }
}