//
// Created by alex on 5/24/25.
//

#include "HierarchyUtils.h"
#include "TransformUtils.h"

namespace Bcg::Hierarchy {
    void UpdateHierarchy() {
        auto view_dirty_local_transforms = Engine::State().view<Dirty<Transform::CachedLocalTransformMatrix<float>>>();

    }
    void UpdateTransforms() {
        auto view = Engine::State().view<Commands::UpdateTransformFromParent>(entt::exclude<Hierarchy::Parent>);
        for (auto entity : view) {
            UpdateChildrenTransformsRecursive(entity);
            Engine::State().remove<Commands::UpdateTransformFromParent>(entity);
        }
    }

    void UpdateTransformsRecursive(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Error("Hierarchy::UpdateTransformsRecursive: Invalid entity ID {}", entity_id);
            return;
        }
        if (!Engine::has<Hierarchy::Parent>(entity_id)) {
            Log::Error("Hierarchy::UpdateTransformsRecursive: Entity {} does not have a parent", entity_id);
            return;
        }
        auto &parent = Engine::State().get<Hierarchy::Parent>(entity_id);
        if (!Engine::has<Transform::CachedWorldTransformMatrix<float>>(parent.parent)) {
            Log::Error("Hierarchy::UpdateTransformsRecursive: Parent entity {} does not have Transform::CachedWorldTransformMatrix", parent.parent);
            return;
        }
        auto &parent_transform = Engine::State().get<Transform::CachedWorldTransformMatrix<float>>(parent.parent);
        auto &cached_local = Engine::State().get<Transform::CachedLocalTransformMatrix<float>>(entity_id);
        Engine::State().emplace_or_replace<Transform::CachedWorldTransformMatrix<float>>(entity_id,
            parent_transform.matrix * cached_local.matrix);

        // Update children
        if (Engine::has<Hierarchy::Children>(entity_id)) {
            auto &children = Engine::State().get<Hierarchy::Children>(entity_id);
            for (const auto &child : children.children) {
                UpdateTransformsRecursive(child);
            }
        }
    }
}

namespace Bcg::Commands {
    void SetParent::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Error("Hierarchy::SetParent: Invalid entity ID {}", entity_id);
            return;
        }
        if (!Engine::valid(parent_id)) {
            Log::Error("Hierarchy::SetParent: Invalid parent ID {}", parent_id);
            return;
        }
        if (Engine::has<Hierarchy::Parent>(entity_id)) {
            auto &current = Engine::State().get<Hierarchy::Parent>(entity_id);
            if (!Engine::has<Hierarchy::Children>(current.parent)) {
                // This means entity blindly follows the parent and we can just update the parent
            }else {
                auto &parents_children = Engine::State().get<Hierarchy::Children>(current.parent);
                auto it = std::find(parents_children.children.begin(), parents_children.children.end(), entity_id);
                if (it != parents_children.children.end()) {
                    Log::Warn("Hierarchy::SetParent: Entity {} is already a child of {}", entity_id, current.parent);
                    return;
                }else {

                }
            }
        }else {
            Engine::State().emplace<Hierarchy::Parent>(entity_id, parent_id);
        }
        Engine::State().emplace_or_replace<Dirty<Hierarchy::Parent>>(entity_id);
    }

    void RemoveParent::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Error("Hierarchy::RemoveParent: Invalid entity ID {}", entity_id);
            return;
        }
        if (!Engine::has<Hierarchy::Parent>(entity_id)) {
            Log::Warn("Hierarchy::RemoveParent: Entity {} does not have a parent", entity_id);
            return;
        }
        Engine::State().remove<Hierarchy::Parent>(entity_id);
        Engine::State().emplace_or_replace<Dirty<Hierarchy::Parent>>(entity_id);
    }

    void AddChild::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Error("Hierarchy::AddChild: Invalid entity ID {}", entity_id);
            return;
        }
        if (!Engine::valid(child_id)) {
            Log::Error("Hierarchy::AddChild: Invalid child ID {}", child_id);
            return;
        }
        if (!Engine::has<Hierarchy::Children>(entity_id)) {
            Engine::State().emplace<Hierarchy::Children>(entity_id);
        }
        auto &children = Engine::State().get<Hierarchy::Children>(entity_id);
        if (std::find(children.children.begin(), children.children.end(), child_id) != children.children.end()) {
            Log::Warn("Hierarchy::AddChild: Entity {} is already a child of {}", child_id, entity_id);
            return;
        }
        children.children.push_back(child_id);
        Engine::State().emplace_or_replace<Dirty<Hierarchy::Children>>(entity_id);
    }

    void RemoveChild::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Error("Hierarchy::AddChild: Invalid entity ID {}", entity_id);
            return;
        }
        if (!Engine::valid(child_id)) {
            Log::Error("Hierarchy::AddChild: Invalid child ID {}", child_id);
            return;
        }
        if (!Engine::has<Hierarchy::Children>(entity_id)) {
            Log::Warn("Hierarchy::RemoveChild: Entity {} does not have children", entity_id);
            return;
        }
        auto &children = Engine::State().get<Hierarchy::Children>(entity_id);
        auto it = std::find(children.children.begin(), children.children.end(), child_id);
        if (it == children.children.end()) {
            Log::Warn("Hierarchy::RemoveChild: Entity {} is not a child of {}", child_id, entity_id);
            return;
        }
        children.children.erase(it);
        if (children.children.empty()) {
            Engine::State().remove<Hierarchy::Children>(entity_id);
        }
        Engine::State().emplace_or_replace<Dirty<Hierarchy::Children>>(entity_id);
    }

    void MarkChildrenTransformDirty::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Error("Hierarchy::MarkChildrenTransformDirty: Invalid entity ID {}", entity_id);
            return;
        }
        if (!Engine::has<Hierarchy::Children>(entity_id)) {
            Log::Warn("Hierarchy::MarkChildrenTransformDirty: Entity {} does not have children", entity_id);
            return;
        }
        auto &children = Engine::State().get<Hierarchy::Children>(entity_id);
        for (const auto &child : children.children) {
            Engine::State().emplace_or_replace<UpdateCachedWorld>(child);
            MarkChildrenTransformDirty mark_child_dirty(child);
            mark_child_dirty.execute();
        }
    }

    void UpdateTransformFromParent::execute() const {
        if (!Engine::valid(entity_id)) {
            Log::Error("Transform::UpdateTransformFromParent: Invalid entity ID {}", entity_id);
            return;
        }
        if (!Engine::has<Hierarchy::Parent>(entity_id)) {
            Log::Error("Transform::UpdateTransformFromParent: Entity {} does not have a parent", entity_id);
            return;
        }
        auto &parent = Engine::State().get<Hierarchy::Parent>(entity_id);
        if (!Engine::has<Transform::CachedWorldTransformMatrix<float>>(parent.parent)) {
            Log::Error("Transform::UpdateTransformFromParent: Parent entity {} does not have Transform::CachedWorldTransformMatrix", parent.parent);
            return;
        }
        auto &parent_transform = Engine::State().get<Transform::CachedWorldTransformMatrix<float>>(parent.parent);
        auto &cached_local = Engine::State().get<Transform::CachedLocalTransformMatrix<float>>(entity_id);
        Engine::State().emplace_or_replace<Transform::CachedWorldTransformMatrix<float>>(entity_id,
            parent_transform.matrix * cached_local.matrix);
    }

}