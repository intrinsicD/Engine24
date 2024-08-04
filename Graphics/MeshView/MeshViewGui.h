//
// Created by alex on 04.08.24.
//

#ifndef ENGINE24_MESHVIEWGUI_H
#define ENGINE24_MESHVIEWGUI_H

#include "MeshView.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void Show(MeshView &view);

    void ShowMeshView(entt::entity entity_id);

}

#endif //ENGINE24_MESHVIEWGUI_H
