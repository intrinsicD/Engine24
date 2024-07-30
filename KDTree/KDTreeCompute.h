//
// Created by alex on 28.07.24.
//

#ifndef ENGINE24_KDTREECOMPUTE_H
#define ENGINE24_KDTREECOMPUTE_H

#include "entt/fwd.hpp"
#include "Properties.h"
#include "Types.h"

namespace Bcg {
    void BuildKDTReeCompute(entt::entity entity_id, Property<Point> points);
}

#endif //ENGINE24_KDTREECOMPUTE_H
