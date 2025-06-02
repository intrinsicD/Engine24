//
// Created by alex on 25.07.24.
//

#include "MeshGui.h"
#include "ImGuiFileDialog.h"
#include "PluginSurfaceMesh.h"
#include "PropertiesGui.h"
#include "Engine.h"

namespace Bcg::Gui {
    void ShowLoadMesh() {
        if (ImGuiFileDialog::Instance()->Display("Load Mesh", ImGuiWindowFlags_NoCollapse, ImVec2(200, 100))) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                // action
                auto mesh = PluginSurfaceMesh::read(filePathName);
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
        if (ImGui::CollapsingHeader(("Vertices #v: " + std::to_string(mesh.n_vertices())).c_str())) {
            ImGui::PushID("Vertices");
            Show("##Vertices", mesh.vprops_);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader(("Halfedges #h: " + std::to_string(mesh.n_halfedges())).c_str())) {
            ImGui::PushID("Halfedges");
            Show("##Halfedges", mesh.hprops_);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader(("Edges #e: " + std::to_string(mesh.n_edges())).c_str())) {
            ImGui::PushID("Edges");
            Show("##Edges",mesh.eprops_);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader(("Faces #f: " + std::to_string(mesh.n_faces())).c_str())) {
            ImGui::PushID("Faces");
            Show("##Faces",mesh.fprops_);
            ImGui::PopID();
        }
    }
}