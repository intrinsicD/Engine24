//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_TRANSFORMGUI_H
#define ENGINE24_TRANSFORMGUI_H

#include "Transform.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void ShowTransform(entt::entity entity_id);

    void Show(Transform &transform);
}

#endif //ENGINE24_TRANSFORMGUI_H
