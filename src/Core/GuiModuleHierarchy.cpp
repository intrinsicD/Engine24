//
// Created by alex on 17.06.25.
//

#include "GuiModuleHierarchy.h"
#include "ParentComponent.h"
#include "ChildrenComponent.h"
#include "Hierarchy.h"
#include "NameComponent.h"
#include "Picker.h"

#include "imgui.h"
#include "entt/entity/registry.hpp"

namespace Bcg {

    GuiModuleHierarchy::GuiModuleHierarchy(entt::registry &registry)
            : GuiModule("Hierarchy"), m_registry(registry) {}

    void GuiModuleHierarchy::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            ImGui::MenuItem(name.c_str(), nullptr, &m_is_window_open);
            ImGui::EndMenu();
        }
    }

    void GuiModuleHierarchy::render_gui() {
        if (!m_is_window_open) {
            return;
        }

        if (ImGui::Begin(name.c_str(), &m_is_window_open)) {
            // We iterate through all entities that are roots (have no parent)
            // and start the recursive drawing process from them.
            auto view = m_registry.view<entt::entity>(entt::exclude<ParentComponent>);
            for (auto entity_id: view) {
                // We also exclude entities that are just children containers but not real objects
                // This is an optional check if you have such entities.
                if (m_registry.valid(entity_id)) {
                    draw_entity_node(entity_id);
                }
            }

            // If the user clicks on the empty space in the window, deselect everything.
            if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered()) {
                auto &picked = m_registry.ctx().get<Picked>();
                picked.entity = {};
            }
        }
        ImGui::End();
    }


    void GuiModuleHierarchy::draw_entity_node(entt::entity entity_id) {
        // Use a tag component for the name, or just the entity ID as a fallback.
        auto *name_comp = m_registry.try_get<NameComponent>(entity_id);
        std::string name = name_comp ? name_comp->name : "Entity " + std::to_string((uint32_t) entity_id);

        // --- Setup Tree Node Flags ---
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
        auto &picked = m_registry.ctx().get<Picked>();
        if (picked.entity.id == entity_id) {
            flags |= ImGuiTreeNodeFlags_Selected;
        }

        auto *children_comp = m_registry.try_get<ChildrenComponent>(entity_id);
        bool has_children = children_comp && !children_comp->children.empty();
        if (!has_children) {
            flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        }

        // --- Draw the Node ---
        bool is_node_open = ImGui::TreeNodeEx((void *) (uint64_t) entity_id, flags, "%s", name.c_str());

        // --- Handle Selection ---
        if (ImGui::IsItemClicked()) {
            picked.entity.id = entity_id;
        }

        // --- Handle Drag-and-Drop ---

        // 1. Drag Source: If we drag this item...
        if (ImGui::BeginDragDropSource()) {
            // Set payload to carry the entity ID.
            ImGui::SetDragDropPayload("HIERARCHY_ENTITY", &entity_id, sizeof(entt::entity));
            ImGui::Text("%s", name.c_str());
            ImGui::EndDragDropSource();
        }

        // 2. Drop Target: If we drop something onto this item...
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload *payload = ImGui::AcceptDragDropPayload("HIERARCHY_ENTITY")) {
                entt::entity payload_entity = *(const entt::entity *) payload->Data;

                // Use our safe, centralized function to perform the re-parenting.
                Hierarchy(m_registry).set_parent(payload_entity, entity_id);
            }
            ImGui::EndDragDropTarget();
        }


        // --- Recurse for Children ---
        if (is_node_open && has_children) {
            for (auto child_id: children_comp->children) {
                if (m_registry.valid(child_id)) {
                    draw_entity_node(child_id);
                }
            }
            if (has_children) { // This check is needed because NoTreePushOnOpen flag
                ImGui::TreePop();
            }
        }
    }
}
