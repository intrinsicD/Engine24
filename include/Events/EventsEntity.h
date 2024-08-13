//
// Created by alex on 18.07.24.
//

#ifndef ENGINE24_EVENTSENTITY_H
#define ENGINE24_EVENTSENTITY_H

#include "entt/fwd.hpp"

namespace Bcg::Events::Entity {
    struct Destroy {
        entt::entity entity_id;
    };

    struct CleanupComponents{
        entt::entity entity_id;
    };

    template<typename Component>
    struct PreAdd {
        entt::entity entity_id;
    };

    template<typename Component>
    struct PostAdd {
        entt::entity entity_id;
    };

    template<typename Component>
    struct PreRemove {
        entt::entity entity_id;
    };

    template<typename Component>
    struct PostRemove {
        entt::entity entity_id;
    };
}

#endif //ENGINE24_EVENTSENTITY_H
