//
// Created by alex on 16.06.25.
//

#include "GuiModuleTransforms.h"
#include "TransformComponent.h"
#include "WorldTransformComponent.h"
#include "ParentComponent.h"
#include "Engine.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "Camera.h"
#include "Renderer.h"
#include "Picker.h"
#include "EntitySelection.h"
#include "entt/entity/registry.hpp"
#include "TransformUtils.h"

#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace Bcg {
    // File-scope gizmo state
    static ImGuizmo::OPERATION g_current_gizmo_operation = ImGuizmo::TRANSLATE;
    static ImGuizmo::MODE g_current_gizmo_mode = ImGuizmo::WORLD;

    GuiModuleTransforms::GuiModuleTransforms(entt::registry &registry, Renderer &renderer)
        : GuiModule("Transforms"), m_registry(registry), m_renderer(renderer) {
    }

    void GuiModuleTransforms::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            ImGui::MenuItem(name.c_str(), nullptr, &m_is_window_open);
            ImGui::EndMenu();
        }
    }

    void GuiModuleTransforms::render_guizmo(Renderer &renderer) {
        const auto &picked = Engine::Context().get<Picked>();
        const auto entity_id = picked.entity.id;

        // Must happen every frame for the gizmo to work.
        // It should be called AFTER the main 3D scene is rendered but BEFORE this GUI is drawn.
        if (entity_id != entt::null && Engine::State().valid(entity_id)) {
            render_gizmo(entity_id, renderer);
        }
    }


    void GuiModuleTransforms::render_gui() {
        const auto &picked = m_registry.ctx().get<Picked>();
        const auto entity_id = picked.entity.id;

        if (!m_is_window_open) {
            return;
        }

        if (ImGui::Begin(name.c_str(), &m_is_window_open)) {
            // Gizmo controls (UI must be inside a window)
            if (ImGui::RadioButton("Translate", g_current_gizmo_operation == ImGuizmo::TRANSLATE)) {
                g_current_gizmo_operation = ImGuizmo::TRANSLATE;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate", g_current_gizmo_operation == ImGuizmo::ROTATE)) {
                g_current_gizmo_operation = ImGuizmo::ROTATE;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Scale", g_current_gizmo_operation == ImGuizmo::SCALE)) {
                g_current_gizmo_operation = ImGuizmo::SCALE;
            }

            if (g_current_gizmo_operation != ImGuizmo::SCALE) {
                if (ImGui::RadioButton("World", g_current_gizmo_mode == ImGuizmo::WORLD)) {
                    g_current_gizmo_mode = ImGuizmo::WORLD;
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Local", g_current_gizmo_mode == ImGuizmo::LOCAL)) {
                    g_current_gizmo_mode = ImGuizmo::LOCAL;
                }
            } else {
                g_current_gizmo_mode = ImGuizmo::LOCAL; // Scale must be local
            }

            ImGui::Separator();

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
            m_registry.emplace_or_replace<DirtyLocalTransform>(entity_id);
        }
    }

    void GuiModuleTransforms::render_gizmo(entt::entity entity_id, Renderer &renderer) {
        // The gizmo needs both a local and a world transform to function correctly.
        if (!Engine::State().all_of<TransformComponent, WorldTransformComponent>(entity_id)) {
            return;
        }

        auto &camera = Engine::Context().get<Camera>();
        const glm::mat4 &view_matrix = camera.view;
        const glm::mat4 &projection_matrix = camera.proj;

        // Get the entity's world transform for the gizmo to manipulate.
        // We make a copy because ImGuizmo will modify it directly.
        glm::mat4 world_transform_matrix = Engine::State().get<WorldTransformComponent>(entity_id).world_transform;

        // --- Gizmo Setup ---
        ImGuizmo::BeginFrame();
        ImGuizmo::Enable(true);
        ImGuizmo::SetOrthographic(Engine::Context().get<Camera>().proj_type == Camera::ProjectionType::ORTHOGRAPHIC);
        // Draw on top of everything
        ImGuizmo::SetDrawlist(ImGui::GetForegroundDrawList(ImGui::GetMainViewport()));

        // Define the viewport area for the gizmo using ImGui viewport (accounts for DPI/position)
        ImGuiViewport *vp = ImGui::GetMainViewport();
        ImGuizmo::SetRect(vp->Pos.x, vp->Pos.y, vp->Size.x, vp->Size.y);

        // Also draw a small view-manipulate gizmo in the lower-right corner

        // --- Render and Manipulate the Gizmo ---
        if (ImGuizmo::Manipulate(glm::value_ptr(view_matrix), glm::value_ptr(projection_matrix),
                                 g_current_gizmo_operation, g_current_gizmo_mode,
                                 glm::value_ptr(world_transform_matrix))) {
            // If the gizmo was used, we need to update the TransformComponent.
            // ImGuizmo modifies the WORLD matrix. We must decompose this back into a LOCAL transform.
            auto &transform = Engine::State().get<TransformComponent>(entity_id);

            // 1. Get the parent's world transform. If no parent, it's the identity matrix.
            glm::mat4 parent_world_transform(1.0f);
            if (auto *parent_comp = Engine::State().try_get<ParentComponent>(entity_id)) {
                if (Engine::State().valid(parent_comp->parent_entity) &&
                    Engine::State().all_of<WorldTransformComponent>(parent_comp->parent_entity)) {
                    parent_world_transform = Engine::State().get<WorldTransformComponent>(
                        parent_comp->parent_entity).world_transform;
                }
            }

            // 2. Calculate the new local matrix by removing the parent's transform.
            // Local = inverse(ParentWorld) * NewWorld
            glm::mat4 new_local_matrix = glm::inverse(parent_world_transform) * world_transform_matrix;

            // 3. Decompose using our helper to match TRS convention, and normalize quaternion
            transform = decompose(new_local_matrix);
            transform.rotation = glm::normalize(transform.rotation);

            // 4. Mark the entity as dirty.
            Engine::State().emplace_or_replace<DirtyLocalTransform>(entity_id);
        }
    }
}
