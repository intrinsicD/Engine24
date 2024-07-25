//
// Created by alex on 25.07.24.
//

#include "MeshGui.h"
#include "ImGuiFileDialog.h"
#include "PluginMesh.h"
#include "PropertiesGui.h"
#include "Engine.h"

namespace Bcg::Gui {
    void ShowLoadMesh() {
        if (ImGuiFileDialog::Instance()->Display("Load Mesh", ImGuiWindowFlags_NoCollapse, ImVec2(200, 100))) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                // action
                auto mesh = PluginMesh::load(filePathName);
            }

            // close
            ImGuiFileDialog::Instance()->Close();
        }
    }

    void ShowSurfaceMesh(entt::entity entity_id) {
        if (Engine::valid(entity_id) && Engine::has<SurfaceMesh>(entity_id)) {
            auto &mesh = Engine::State().get<SurfaceMesh>(entity_id);
            Show(mesh);
        }
    }

    void Show(SurfaceMesh &mesh) {
        if (ImGui::CollapsingHeader("Vertices")) {
            ImGui::PushID("Vertices");
            Show(mesh.vprops_);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader("Halfedges")) {
            ImGui::PushID("Halfedges");
            Show(mesh.hprops_);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader("Edges")) {
            ImGui::PushID("Edges");
            Show(mesh.eprops_);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader("Faces")) {
            ImGui::PushID("Faces");
            Show(mesh.fprops_);
            ImGui::PopID();
        }
    }
}