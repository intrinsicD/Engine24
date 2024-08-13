//
// Created by alex on 02.08.24.
//

#ifndef ENGINE24_SPHEREVIEWGUI_H
#define ENGINE24_SPHEREVIEWGUI_H

#include "SphereView.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void Show(SphereView &view);

    void ShowSphereView(entt::entity entity_id);
}

#endif //ENGINE24_SPHEREVIEWGUI_H
