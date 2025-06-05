//
// Created by alex on 08.11.24.
//

#ifndef ENGINE24_KDTREEGUI_H
#define ENGINE24_KDTREEGUI_H

#include "Cuda/BVHCuda.h"

namespace Bcg::Gui {
    void ShowKDTree(entt::entity entity_id);

    void Show(cuda::BVHCuda &kdtree);
}

#endif //ENGINE24_KDTREEGUI_H
