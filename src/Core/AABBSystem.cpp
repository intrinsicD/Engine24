//
// Created by alex on 16.06.25.
//

#include "AABBSystem.h"
#include "AABBComponents.h"
#include "AABBUtils.h"
#include "WorldTransformComponent.h"
#include "entt/entity/registry.hpp"

namespace Bcg {
    void UpdateWorldAABBSystem(entt::registry &registry){
        auto view = registry.view<LocalAABB, DirtyWorldTransform>();
        for (const auto &entity : view) {
            auto &local = view.get<LocalAABB>(entity);
            WorldAABB &world = registry.get_or_emplace<WorldAABB>(entity);
            if (registry.all_of<WorldTransformComponent>(entity)) {
                const auto &transform = registry.get<WorldTransformComponent>(entity);
                world.aabb = apply_transform(local.aabb, transform.world_transform);
            }else {
                world.aabb = local.aabb;
            }
            registry.emplace_or_replace<DirtyWorldAABB>(entity);
        }
    }

    void ClearWorldAABBDirtyTags(entt::registry &registry) {
        registry.clear<DirtyWorldAABB>();
    }

}