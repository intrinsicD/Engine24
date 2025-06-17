//
// Created by alex on 16.06.25.
//

#include "GuiModuleTransforms.h"
#include "TransformComponent.h"
#include "TransformDirty.h"
#include "WorldTransformComponent.h"
#include "ParentComponent.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "Camera.h"
#include "Renderer.h"
#include "EntitySelection.h"
#include "entt/entity/registry.hpp"

#include <glm/gtx/matrix_decompose.hpp>

namespace Bcg {
    GuiModuleTransforms::GuiModuleTransforms(entt::registry &registry, Renderer &renderer,
                                             EntitySelection &entity_selection)
            : GuiModule("Transforms"), m_registry(registry), m_renderer(renderer), m_entity_selection(entity_selection) {}

    void GuiModuleTransforms::render_menu() {
        if (ImGui::BeginMenu("Engine")) {
            ImGui::MenuItem(name.c_str(), nullptr, &m_is_window_open);
            ImGui::EndMenu();
        }
    }

    void GuiModuleTransforms::render_gui() {
        auto entity_id = m_entity_selection.get_selected_entity();

        // Must happen every frame for the gizmo to work.
        // It should be called AFTER the main 3D scene is rendered but BEFORE this GUI is drawn.
        if (entity_id != entt::null && m_registry.valid(entity_id)) {
            render_gizmo(entity_id);
        }

        if (!m_is_window_open) {
            return;
        }

        if (ImGui::Begin(name.c_str(), &m_is_window_open)) {
            if (entity_id != entt::null && m_registry.valid(entity_id)) {
                render_inspector_panel(entity_id);
            } else {
                ImGui::Text("No entity selected.");
            }
        }
        ImGui::End();
    }

    // --- Private Helper Implementations ---

    void GuiModuleTransforms::render_inspector_panel(entt::entity entity_id) {
        if (!m_registry.all_of<TransformComponent>(entity_id)) {
            ImGui::Text("Selected entity does not have a TransformComponent.");
            return;
        }

        auto &transform = m_registry.get<TransformComponent>(entity_id);
        bool modified = false;

        // Position
        if (ImGui::DragFloat3("Position", glm::value_ptr(transform.position), 0.1f)) {
            modified = true;
        }

        // Rotation (displayed as Euler angles for user-friendliness)
        glm::vec3 euler_angles = glm::degrees(glm::eulerAngles(transform.rotation));
        if (ImGui::DragFloat3("Rotation", glm::value_ptr(euler_angles), 1.0f)) {
            transform.rotation = glm::quat(glm::radians(euler_angles));
            modified = true;
        }

        // Scale
        if (ImGui::DragFloat3("Scale", glm::value_ptr(transform.scale), 0.1f)) {
            modified = true;
        }

        // If any value was changed, mark the entity's transform as dirty.
        if (modified) {
            m_registry.emplace_or_replace<TransformDirty>(entity_id);
        }
    }

    void GuiModuleTransforms::render_gizmo(entt::entity entity_id) {
        // The gizmo needs both a local and a world transform to function correctly.
        if (!m_registry.all_of<TransformComponent, WorldTransformComponent>(entity_id)) {
            return;
        }

        auto &camera = m_registry.ctx().get<Camera>();
        const glm::mat4 &view_matrix = camera.view;
        const glm::mat4 &projection_matrix = camera.proj;

        // Get the entity's world transform for the gizmo to manipulate.
        // We make a copy because ImGuizmo will modify it directly.
        glm::mat4 world_transform_matrix = m_registry.get<WorldTransformComponent>(entity_id).world_transform;

        // --- Gizmo Setup ---
        ImGuizmo::SetOrthographic(false); // Assuming perspective projection
        ImGuizmo::SetDrawlist();

        // Define the viewport area for the gizmo
        const auto &viewport = m_renderer.get_viewport().vec();
        ImGuizmo::SetRect(viewport[0], viewport[1], viewport[2], viewport[3]);

        static ImGuizmo::OPERATION m_current_gizmo_operation = ImGuizmo::TRANSLATE;
        static ImGuizmo::MODE m_current_gizmo_mode = ImGuizmo::WORLD;

        // --- Gizmo Controls UI (can be placed in the inspector window or a toolbar) ---
        if (ImGui::RadioButton("Translate", m_current_gizmo_operation == ImGuizmo::TRANSLATE)) {
            m_current_gizmo_operation = ImGuizmo::TRANSLATE;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", m_current_gizmo_operation == ImGuizmo::ROTATE)) {
            m_current_gizmo_operation = ImGuizmo::ROTATE;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", m_current_gizmo_operation == ImGuizmo::SCALE)) {
            m_current_gizmo_operation = ImGuizmo::SCALE;
        }

        // World vs Local space for Gizmo
        if (m_current_gizmo_operation != ImGuizmo::SCALE) {
            if (ImGui::RadioButton("World", m_current_gizmo_mode == ImGuizmo::WORLD)) {
                m_current_gizmo_mode = ImGuizmo::WORLD;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Local", m_current_gizmo_mode == ImGuizmo::LOCAL)) {
                m_current_gizmo_mode = ImGuizmo::LOCAL;
            }
        } else {
            // Scale must be in local space
            m_current_gizmo_mode = ImGuizmo::LOCAL;
        }

        ImGui::Separator();

        // --- Render and Manipulate the Gizmo ---
        if (ImGuizmo::Manipulate(glm::value_ptr(view_matrix), glm::value_ptr(projection_matrix),
                                 m_current_gizmo_operation, m_current_gizmo_mode,
                                 glm::value_ptr(world_transform_matrix))) {

            // If the gizmo was used, we need to update the TransformComponent.
            // ImGuizmo modifies the WORLD matrix. We must decompose this back into a LOCAL transform.
            auto &transform = m_registry.get<TransformComponent>(entity_id);

            // 1. Get the parent's world transform. If no parent, it's the identity matrix.
            glm::mat4 parent_world_transform(1.0f);
            if (auto *parent_comp = m_registry.try_get<ParentComponent>(entity_id)) {
                if (m_registry.valid(parent_comp->parent_entity) &&
                        m_registry.all_of<WorldTransformComponent>(parent_comp->parent_entity)) {
                    parent_world_transform = m_registry.get<WorldTransformComponent>(
                            parent_comp->parent_entity).world_transform;
                }
            }

            // 2. Calculate the new local matrix by removing the parent's transform.
            // Local = inverse(ParentWorld) * NewWorld
            glm::mat4 new_local_matrix = glm::inverse(parent_world_transform) * world_transform_matrix;

            // 3. Decompose the new local matrix back into PRS components.
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(new_local_matrix, transform.scale, transform.rotation, transform.position, skew,
                           perspective);

            // 4. Mark the entity as dirty.
            m_registry.emplace_or_replace<TransformDirty>(entity_id);
        }
    }
}