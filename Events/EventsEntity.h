//
// Created by alex on 18.07.24.
//

#ifndef ENGINE24_EVENTSENTITY_H
#define ENGINE24_EVENTSENTITY_H

#include "entt/fwd.hpp"

namespace Bcg::Events::Entity {
    template<typename Component>
    struct PreAdd {
        entt::entity entity;
    };

    template<typename Component>
    struct PostAdd {
        entt::entity entity;
    };

    template<typename Component>
    struct PreRemove {
        entt::entity entity;
    };

    template<typename Component>
    struct PostRemove {
        entt::entity entity;
    };
}

#endif //ENGINE24_EVENTSENTITY_H
