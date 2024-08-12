//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_HIERARCHY_H
#define ENGINE24_HIERARCHY_H

#include "entt/entt.hpp"

namespace Bcg {
    struct Hierarchy {
        entt::entity parent = entt::null;
        std::vector<entt::entity> children;
        std::vector<entt::entity> overlays;

        bool has_child(entt::entity child) {
            return std::find(children.begin(), children.end(), child) != children.end();
        }

        bool has_overlay(entt::entity overlay) {
            return std::find(overlays.begin(), overlays.end(), overlay) != overlays.end();
        }
    };
}

#endif //ENGINE24_HIERARCHY_H
