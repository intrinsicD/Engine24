//
// Created by alex on 03.07.24.
//

#ifndef ENGINE24_PICKER_H
#define ENGINE24_PICKER_H

#include "entt/fwd.hpp"
#include "MatVec.h"
#include "Plugin.h"
#include "GuiUtils.h"
#include "CoordinateSystems.h"

namespace Bcg {
    struct Picked {
        struct Entity {
            entt::entity id;
            bool is_background = true;

            operator bool() { return !is_background; }

            unsigned int vertex_idx = -1;
            unsigned int edge_idx = -1;
            unsigned int face_idx = -1;
        } entity;

        Points spaces;
    };
}

#endif //ENGINE24_PICKER_H
