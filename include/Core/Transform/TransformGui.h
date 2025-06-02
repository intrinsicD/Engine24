//
// Created by alex on 30.10.24.
//

#ifndef ENGINE24_TRANSFORMGUI_H
#define ENGINE24_TRANSFORMGUI_H

#include "Transform.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    bool Show(Transform &transform);

    bool Show(TransformParameters &t_params);

    void Show(const glm::mat4 &mat);

    bool ShowGuizmo(glm::mat4 &mat, glm::mat4 &delta, bool &is_scaling);

    bool ShowTransform(entt::entity entity_id);
}

#endif //ENGINE24_TRANSFORMGUI_H
