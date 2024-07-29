//
// Created by alex on 28.07.24.
//

#ifndef ENGINE24_PLUGINHIERARCHY_H
#define ENGINE24_PLUGINHIERARCHY_H

#include "Plugin.h"
#include "Hierarchy.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class PluginHierarchy : public Plugin {
    public:
        PluginHierarchy() : Plugin("Hierarchy") {}

        ~PluginHierarchy() override = default;

        static void attach_child(entt::entity parent, entt::entity child);

        static bool detach_child(entt::entity parent, entt::entity child);

        static void attach_overlay(entt::entity parent, entt::entity overlay);

        static bool detach_overlay(entt::entity parent, entt::entity overlay);

        static void detach_children(entt::entity parent);

        static void detach_overlays(entt::entity parent);

        static void detach_all(entt::entity parent);

        static void attach_parent(entt::entity child, entt::entity new_parent);

        static bool detach_parent(entt::entity child);

        static void mark_transforms_dirty(entt::entity entity);

        static void update_transforms(entt::entity entity);

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };
}

#endif //ENGINE24_PLUGINHIERARCHY_H
