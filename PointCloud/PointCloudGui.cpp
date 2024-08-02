//
// Created by alex on 30.07.24.
//


#include "PointCloudGui.h"
#include "ImGuiFileDialog.h"
#include "PluginPointCloud.h"
#include "PropertiesGui.h"
#include "Engine.h"

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
        if (Engine::valid(entity_id) && Engine::has<PointCloud>(entity_id)) {
            auto &pc = Engine::State().get<PointCloud>(entity_id);
            Show(pc);
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