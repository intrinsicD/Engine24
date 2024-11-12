//
// Created by alex on 28.07.24.
//

#ifndef ENGINE24_AABBGUI_H
#define ENGINE24_AABBGUI_H

#include "AABB.h"
#include "Pool.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void Show(const PoolHandle<AABB> &h_aabb);

    void Show(const AABB &aabb);

    void Show(entt::entity entity_id);
}

#endif //ENGINE24_AABBGUI_H
