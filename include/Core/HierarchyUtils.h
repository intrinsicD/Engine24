//
// Created by alex on 22.05.25.
//

#ifndef ENGINE24_HIERARCHYUTILS_H
#define ENGINE24_HIERARCHYUTILS_H

#include "Hierarchy.h"
#include "Transform.h"
#include "Engine.h"
#include "Command.h"
#include "Logger.h"

namespace Bcg::Hierarchy {
    struct NeedUpdateFromParentWorld{

    };

    void UpdateHierarchy();

    void UpdateTransforms();

    void UpdateChildrenTransformsRecursive(entt::entity entity_id);
}

namespace Bcg::Commands {
    struct SetParent : public AbstractCommand {
        SetParent(entt::entity entity_id, entt::entity parent_id)
            : AbstractCommand("SetParent"), entity_id(entity_id), parent_id(parent_id) {
        }

        void execute() const override;

        entt::entity entity_id;
        entt::entity parent_id;
    };

    struct RemoveParent : public AbstractCommand {
        RemoveParent(entt::entity entity_id)
            : AbstractCommand("RemoveParent"), entity_id(entity_id) {
        }

        void execute() const override;

        entt::entity entity_id;
    };

    struct AddChild : public AbstractCommand {
        AddChild(entt::entity entity_id, entt::entity child_id)
            : AbstractCommand("AddChild"), entity_id(entity_id), child_id(child_id) {
        }

        void execute() const override;

        entt::entity entity_id;
        entt::entity child_id;
    };

    struct RemoveChild : public AbstractCommand {
        RemoveChild(entt::entity entity_id, entt::entity child_id)
            : AbstractCommand("RemoveChild"), entity_id(entity_id), child_id(child_id) {
        }

        void execute() const override;

        entt::entity entity_id;
        entt::entity child_id;
    };

    struct MarkChildrenTransformDirty : public AbstractCommand {
        MarkChildrenTransformDirty(entt::entity entity_id)
            : AbstractCommand("MarkChildrenTransformDirty"), entity_id(entity_id) {
        }

        void execute() const override;

        entt::entity entity_id;
    };

    struct UpdateTransformFromParent : public AbstractCommand {
        UpdateTransformFromParent(entt::entity entity_id)
            : AbstractCommand("UpdateTransformFromParent"), entity_id(entity_id) {
        }

        void execute() const override;

        entt::entity entity_id;
    };
}
#endif //ENGINE24_HIERARCHYUTILS_H
