//
// Created by alex on 26.11.24.
//

#ifndef ENGINE24_MESHCOMPONENT_H
#define ENGINE24_MESHCOMPONENT_H

#include "PoolHandle.h"
#include "SurfaceMesh.h"
#include <vector>

namespace Bcg {
    struct MeshComponent {
        std::vector<PoolHandle<SurfaceMesh>> meshes;
        PoolHandle<SurfaceMesh> current_mesh;
    };
}

#endif //ENGINE24_MESHCOMPONENT_H
