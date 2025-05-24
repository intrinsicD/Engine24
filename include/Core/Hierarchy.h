//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_HIERARCHY_H
#define ENGINE24_HIERARCHY_H

#include "entt/entt.hpp"
#include <vector>

namespace Bcg::Hierarchy {
    struct Parent {
        entt::entity parent = entt::null;
    };

    struct Children {
        std::vector<entt::entity> children;
    };
}

#endif //ENGINE24_HIERARCHY_H
