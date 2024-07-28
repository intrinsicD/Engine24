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
    };
}

#endif //ENGINE24_HIERARCHY_H
