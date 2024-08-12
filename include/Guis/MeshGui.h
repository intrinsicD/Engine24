//
// Created by alex on 25.07.24.
//

#ifndef ENGINE24_MESHGUI_H
#define ENGINE24_MESHGUI_H

#include "SurfaceMesh.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void ShowLoadMesh();

    void ShowSurfaceMesh(entt::entity entity_id);

    void Show(SurfaceMesh &mesh);
}

#endif //ENGINE24_MESHGUI_H
