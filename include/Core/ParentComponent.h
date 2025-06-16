//
// Created by alex on 16.06.25.
//

#ifndef ENGINE24_PARENTCOMPONENT_H
#define ENGINE24_PARENTCOMPONENT_H

#include "entt/entity/entity.hpp"

namespace Bcg{
    struct ParentComponent{
        entt::entity parent_entity = entt::null;
    };
}

#endif //ENGINE24_PARENTCOMPONENT_H
