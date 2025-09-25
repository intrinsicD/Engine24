//
// Created by alex on 17.06.25.
//

#include "Hierarchy.h"
#include "ParentComponent.h"
#include "ChildrenComponent.h"
#include "../../include/Core/Transform/WorldTransformComponent.h"
#include "TransformComponent.h"
#include "TransformSystem.h"
#include "entt/entity/registry.hpp"
#include <glm/gtx/matrix_decompose.hpp>

namespace Bcg{
    Hierarchy::Hierarchy(entt::registry &registry) : m_registry(registry) {}

    void Hierarchy::set_parent(entt::entity child_id, entt::entity parent_id) {
// Ensure entities are valid
        if (!m_registry.valid(child_id)) return;
        if (parent_id != entt::null && !m_registry.valid(parent_id)) return;

        // --- 1. Get the child's current world transform BEFORE any changes ---
        // We must update the transform system first to ensure we have the latest matrix.
        // This is a rare case where one system might trigger another directly.
        UpdateTransformSystem(m_registry);
        const glm::mat4 child_world_transform = m_registry.get<WorldTransformComponent>(child_id).world_transform;

        // --- 2. Detach from the old parent ---
        if (auto* old_parent_comp = m_registry.try_get<ParentComponent>(child_id)) {
            entt::entity old_parent_entity = old_parent_comp->parent_entity;
            if (m_registry.valid(old_parent_entity)) {
                auto& old_parent_children = m_registry.get<ChildrenComponent>(old_parent_entity);
                // Erase the child from the old parent's children list
                std::erase(old_parent_children.children, child_id);
            }
            // Remove the ParentComponent from the child
            m_registry.remove<ParentComponent>(child_id);
        }

        // --- 3. Attach to the new parent ---
        if (parent_id != entt::null) {
            // Add ParentComponent to the child, pointing to the new parent
            m_registry.emplace<ParentComponent>(child_id, parent_id);

            // Add the child to the new parent's ChildrenComponent
            // Get-or-emplace ensures the component exists
            auto& parent_id_children = m_registry.get_or_emplace<ChildrenComponent>(parent_id);
            parent_id_children.children.push_back(child_id);
        }

        // --- 4. CRITICAL: Update the child's local transform ---
        auto& child_local_transform = m_registry.get<TransformComponent>(child_id);

        if (parent_id != entt::null) {
            const auto& parent_id_world = m_registry.get<WorldTransformComponent>(parent_id).world_transform;
            glm::mat4 new_local_matrix = glm::inverse(parent_id_world) * child_world_transform;

            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(new_local_matrix,
                           child_local_transform.scale,
                           child_local_transform.rotation,
                           child_local_transform.position,
                           skew, perspective);
        } else {
            // If un-parenting, the new local transform is just the old world transform
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(child_world_transform,
                           child_local_transform.scale,
                           child_local_transform.rotation,
                           child_local_transform.position,
                           skew, perspective);
        }

        // --- 5. Mark the child as dirty to propagate changes down its own hierarchy ---
        m_registry.emplace_or_replace<DirtyWorldTransform>(child_id);
    }

    void Hierarchy::destroy_entity(entt::entity entity_id) {
        if (!m_registry.valid(entity_id)) return;

        // First, detach from parent so its ChildrenComponent is cleaned up.
        if (auto* parent_comp = m_registry.try_get<ParentComponent>(entity_id)) {
            set_parent(entity_id, entt::null);
        }

        // Recursively destroy all children
        if (auto* children_comp = m_registry.try_get<ChildrenComponent>(entity_id)) {
            // Make a copy, as the component's list will be modified during iteration
            auto children_copy = children_comp->children;
            for (auto child : children_copy) {
                destroy_entity(child);
            }
        }
    }
}