//
// Created by alex on 28.07.24.
//

#ifndef ENGINE24_AABBGUI_H
#define ENGINE24_AABBGUI_H

#include "AABBStruct.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void Show(const AABB &aabb);

    void Show(entt::entity entity_id);
}

#endif //ENGINE24_AABBGUI_H
