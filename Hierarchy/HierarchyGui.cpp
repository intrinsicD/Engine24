//
// Created by alex on 29.07.24.
//

#include "HierarchyGui.h"
#include "Engine.h"
#include "imgui.h"
#include "PluginHierarchy.h"
#include "GuiUtils.h"

namespace Bcg ::Gui {
    void Show(const std::vector<entt::entity> &children) {
        for (auto child: children) {
            if (ImGui::TreeNode(std::to_string(static_cast<unsigned int>(child)).c_str())) {
                ShowHierarchy(child);
                ImGui::TreePop();
            }
        }
    }

    void Show(const Hierarchy &hierarchy) {
        ImGui::Text("Parent: %d", hierarchy.parent);
        if (ImGui::CollapsingHeader(("Children #" + std::to_string(hierarchy.children.size())).c_str())) {
            Show(hierarchy.children);
        }
        if (ImGui::CollapsingHeader(("Overlays #" + std::to_string(hierarchy.overlays.size())).c_str())) {
            Show(hierarchy.overlays);
        }
    }

    void Edit(entt::entity owner, Hierarchy &hierarchy) {
        //Combobox of all entities which parent can be chosen from
        if (ImGui::BeginCombo("Parent", std::to_string(static_cast<unsigned int>(hierarchy.parent)).c_str())) {
            for (auto entity: Engine::State().view<entt::entity>()) {
                if (entity == owner) continue;
                bool is_selected = hierarchy.parent == entity;
                if (ImGui::Selectable(std::to_string(static_cast<unsigned int>(entity)).c_str(), is_selected)) {
                    PluginHierarchy::set_parent(owner, entity);
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::Button("Clear Parent")) {
            PluginHierarchy::remove_parent(owner);
        }
    }

    void ShowHierarchy(entt::entity entity) {
        if (Engine::State().all_of<Hierarchy>(entity)) {
            static bool edit = false;
            ImGui::Checkbox("Edit", &edit);
            if (edit) {
                Edit(entity, Engine::State().get<Hierarchy>(entity));
            } else {
                Show(Engine::State().get<Hierarchy>(entity));
            }
        }
    }
}
