//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_TRANSFORMGUI_H
#define ENGINE24_TRANSFORMGUI_H

#include "Transform.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void Show(RigidTransform &transform);

    bool Edit(RigidTransform &transform);

    bool Show(Transform &transform);

    bool ShowGuizmo(const RigidTransform &transform, RigidTransform &delta, bool &is_scaling);

    bool ShowLocal(RigidTransform &transform);

    void ShowWorld(const RigidTransform &transform);
}

#endif //ENGINE24_TRANSFORMGUI_H
