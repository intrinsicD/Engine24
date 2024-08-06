//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_EVENTSPICKER_H
#define ENGINE24_EVENTSPICKER_H

#include "entt/fwd.hpp"

namespace Bcg::Events {
    struct PickedEntity {
        entt::entity entity_id;
    };

    struct PickedVertex {
        entt::entity entity_id;
        unsigned int idx;
    };

    struct PickedBackgound{

    };
}

#endif //ENGINE24_EVENTSPICKER_H
