//
// Created by alex on 04.07.24.
//

#include "Material.h"
#include "imgui.h"
#include "../GLUtils.h"

namespace Bcg::Gui {
    void Show(Material &material) {
        ImGui::Text("vao: %u", material.vao);
        ImGui::Text("program: %u", material.program);
        ImGui::Text("offset: %u", material.offset);
        ImGui::Text("size: %u", material.size);

        ImGui::Separator();

        for (auto &attribute: material.attributes) {
            Show(attribute);
        }

        if (ImGui::CollapsingHeader("Textures")) {
            for (auto &item: material.textures) {
                ImGui::Text("%s: %u", item.first.c_str(), item.second);
            }
        }
    }

    void Show(Material::Attribute &attribute) {
        ImGui::Text("%s", attribute.name.c_str());
        ImGui::Text("%u", attribute.index);
        ImGui::Text("%u", attribute.size);
        ImGui::Text("%s", glName(attribute.type));
        ImGui::Text((attribute.normalized ? "normalized" : "unnormalized"));
        ImGui::Text("%u", attribute.stride);
        ImGui::Text("%p", attribute.pointer);
    }

    void Show(MeshMaterial &material) {
        ImGui::PushID("MeshMaterial");
        ImGui::ColorEdit3("BaseColor", &material.base_color[0]);
        if (ImGui::CollapsingHeader("Base")) {
            Show(static_cast<Material &>(material));
        }
        ImGui::PopID();
    }

    void Show(GraphMaterial &material) {
        ImGui::PushID("GraphMaterial");
        ImGui::ColorEdit3("BaseColor", &material.base_color[0]);
        if (ImGui::CollapsingHeader("Base")) {
            Show(static_cast<Material &>(material));
        }
        ImGui::PopID();
    }

    void Show(PointCloudMaterial &material) {
        ImGui::PushID("PointCloudMaterial");
        ImGui::ColorEdit3("BaseColor", &material.base_color[0]);
        if (ImGui::CollapsingHeader("Base")) {
            Show(static_cast<Material &>(material));
        }
        ImGui::PopID();
    }
}