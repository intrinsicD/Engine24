//
// Created by alex on 16.06.25.
//

#ifndef ENGINE24_CHILDRENCOMPONENT_H
#define ENGINE24_CHILDRENCOMPONENT_H

#include "entt/fwd.hpp"
#include <vector>

namespace Bcg{
    struct ChildrenComponent{
        std::vector<entt::entity> children;
    };
}

#endif //ENGINE24_CHILDRENCOMPONENT_H
