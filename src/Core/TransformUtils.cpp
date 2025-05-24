//
// Created by alex on 5/24/25.
//

#include "TransformUtils.h"
#include "HierarchyUtils.h"
#include "Engine.h"
#include "Logger.h"

namespace Bcg::Transform {
    void UpdateCachedLocal() {
        auto view = Engine::State().view<Parameters<float>, Dirty<Parameters<float> > >();
        for (auto entity: view) {
            auto &params = Engine::State().get<Parameters<float> >(entity);
            Engine::State().emplace_or_replace<CachedLocalTransformMatrix<float> >(entity, GetTransform(params));
            Engine::State().emplace_or_replace<Dirty<CachedLocalTransformMatrix<float> > >(entity);
            Engine::State().remove<Dirty<Parameters<float> > >(entity);
        }
    }

    void MarkChildrenDirtyRecursively(entt::entity entity) {
        if (!Engine::has<Hierarchy::Children>(entity)) {
            return;
        }

        auto &children = Engine::State().get<Hierarchy::Children>(entity);
        for (const auto &child: children.children) {
            Engine::State().emplace_or_replace<Hierarchy::NeedUpdateFromParentWorld>(child);
            MarkChildrenDirtyRecursively(child);
        }
    }

    void UpdateChildrenRecursively(entt::entity entity) {
        if (!Engine::has<Hierarchy::Children>(entity)) {
            return;
        }
        if (!Engine::has<Hierarchy::NeedUpdateFromParentWorld>(entity)) {
            return;
        }
        if (!Engine::has<CachedWorldTransformMatrix<float> >(entity)) {
            Log::Error("Transform::UpdateChildrenRecursively: Entity {} does not have CachedWorldTransformMatrix", entity);
            return;
        }

        auto &chached_world = Engine::State().get<CachedWorldTransformMatrix<float> >(entity);

        auto &children = Engine::State().get<Hierarchy::Children>(entity);
        for (const auto &child: children.children) {
            auto &child_chached_local = Engine::State().get<CachedLocalTransformMatrix<float> >(child);
            Engine::State().emplace_or_replace<CachedWorldTransformMatrix<float> >(child,
                chached_world.matrix * child_chached_local.matrix);
            Engine::State().remove<Hierarchy::NeedUpdateFromParentWorld>(child);
            UpdateChildrenRecursively(child);
        }
    }

    void UpdateCachedWorld() {
        auto view = Engine::State().view<CachedLocalTransformMatrix<float>, Dirty<CachedLocalTransformMatrix<float> > >();
        for (auto entity: view) {
            if (Engine::has<Hierarchy::Parent>(entity)) {

            }
            if (Engine::has<Hierarchy::Children>(entity)) {
                MarkChildrenDirtyRecursively(entity);
            }else {

            }
            auto &cached_local = Engine::State().get<CachedLocalTransformMatrix<float> >(entity);
            Engine::State().emplace_or_replace<CachedWorldTransformMatrix<float> >(entity, cached_local);
            Engine::State().emplace_or_replace<Dirty<CachedWorldTransformMatrix<float> > >(entity);
            Engine::State().remove<Dirty<CachedLocalTransformMatrix<float> > >(entity);
        }

        auto view_dirty_world_no_parent = Engine::State().view<Dirty<CachedWorldTransformMatrix<float> >>(entt::exclude<Hierarchy::NeedUpdateFromParentWorld>);
        for (auto entity: view_dirty_world_no_parent) {

        }
    }

    void Update() {
        UpdateCachedLocal();
        UpdateCachedWorld();

        // Update transforms based on hierarchy
        auto view = Engine::State().view<Dirty<CachedWorldTransformMatrix<float>>>(entt::exclude<Hierarchy::NeedUpdateFromParentWorld>);
        for (auto entity: view) {
            UpdateChildrenRecursively(entity);
        }
    }
}

namespace Bcg::Commands {
    void SetParameters::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Error("Transform::SetParameters: Invalid entity ID {}", entity_id);
            return;
        }
        Engine::State().emplace_or_replace<Transform::Parameters<float> >(entity_id, params);
        Engine::State().emplace_or_replace<Transform::CachedLocalTransformMatrix<float> >(
            entity_id, Transform::GetTransform(params));
        Engine::State().emplace_or_replace<Dirty<Transform::CachedLocalTransformMatrix<float> > >(entity_id);
    }

    void UpdateCachedWorld::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Error("Transform::UpdateCachedLocal: Invalid entity ID {}", entity_id);
            return;
        }
        if (!Engine::has<Dirty<Transform::CachedLocalTransformMatrix<float> > >(entity_id)) {
            Log::Error(
                "Transform::UpdateCachedWorld: Entity {} does not have Dirty<Transform::CachedLocalTransformMatrix<float>> component",
                entity_id);
            return;
        }
        auto &cached_local = Engine::State().get<Transform::CachedLocalTransformMatrix<float> >(entity_id);
        Engine::State().emplace_or_replace<Transform::CachedWorldTransformMatrix<float> >(entity_id, cached_local);
        Engine::State().emplace_or_replace<Dirty<Transform::CachedWorldTransformMatrix<float> > >(entity_id);
    }
}
