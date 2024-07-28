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

        static void add_child(entt::entity parent, entt::entity child);

        static bool remove_child(entt::entity parent, entt::entity child);

        static void add_overlay(entt::entity parent, entt::entity overlay);

        static void remove_overlay(entt::entity parent, entt::entity overlay);

        static void clear_children(entt::entity parent);

        static void clear_overlays(entt::entity parent);

        static void clear(entt::entity parent);

        static void set_parent(entt::entity child, entt::entity new_parent);

        static void remove_parent(entt::entity child);

        static void update_transforms(entt::entity parent);

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
