//
// Created by alex on 16.06.25.
//

#include "TransformSystem.h"
#include "TransformComponent.h"
#include "../../include/Core/Transform/WorldTransformComponent.h"
#include "ParentComponent.h"
#include "ChildrenComponent.h"
#include "entt/entity/registry.hpp"
#include <unordered_set>
#include <queue>

namespace Bcg {
    void UpdateTransformSystem(entt::registry &registry) {
        // --- PHASE 1: Identify all unique roots of dirty hierarchies ---

        // Using a set to ensure we process each dirty hierarchy only once,
        // even if multiple children in the same hierarchy were dirtied.
        std::unordered_set<entt::entity> dirty_roots;

        auto dirty_view = registry.view<DirtyLocalTransform>();
        for (auto entity: dirty_view) {
            auto current = entity;
            // Traverse up the hierarchy until we find the root parent.
            // The root is an entity that has no ParentComponent.
            while (auto *parent_comp = registry.try_get<ParentComponent>(current)) {
                current = parent_comp->parent_entity;
            }
            dirty_roots.insert(current);
        }

        // --- PHASE 2: Update each dirty hierarchy from its root ---

        std::queue<entt::entity> update_queue;

        for (const auto root: dirty_roots) {
            // Start a breadth-first traversal from the root.
            // This guarantees that parents are always processed before their children.
            update_queue.push(root);

            while (!update_queue.empty()) {
                auto current_entity = update_queue.front();
                update_queue.pop();

                // 1. Calculate the final world matrix
                const auto &local_transform = registry.get<TransformComponent>(current_entity);
                glm::mat4 parent_world_matrix = glm::mat4(1.0f);

                if (auto *parent_comp = registry.try_get<ParentComponent>(current_entity)) {
                    // The parent's WorldTransformComponent MUST be up-to-date here
                    // because of the breadth-first processing order.
                    parent_world_matrix = registry.get<WorldTransformComponent>(
                            parent_comp->parent_entity).world_transform;
                }

                glm::mat4 final_world_matrix = parent_world_matrix * local_transform.matrix();

                // 2. Update or add the WorldTransformComponent
                registry.emplace_or_replace<WorldTransformComponent>(current_entity, final_world_matrix);

                // 3. Mark the entity as clean
                registry.remove<DirtyLocalTransform>(current_entity);
                registry.emplace_or_replace<DirtyWorldTransform>(current_entity);

                // 4. Add all direct children to the queue for the next level of processing
                if (auto *children_comp = registry.try_get<ChildrenComponent>(current_entity)) {
                    for (const auto child: children_comp->children) {
                        update_queue.push(child);
                    }
                }
            }
        }
    }
    void ClearTransformDirtyTags(entt::registry &registry) {
        registry.clear<DirtyLocalTransform>();
        registry.clear<DirtyWorldTransform>();
    }
}