//
// Created by alex on 21.06.24.
//

#ifndef ENGINE24_SURFACEMESHCOMPUTE_H
#define ENGINE24_SURFACEMESHCOMPUTE_H

#include "SurfaceMesh.h"
#include "entt/fwd.hpp"

namespace Bcg {
    VertexProperty<Vector<float, 3>> ComputeSurfaceMeshVertexNormals(entt::entity entity_id);

    FaceProperty<Vector<float, 3>> ComputeFaceNormals(entt::entity entity_id, SurfaceMesh &mesh);

    void PT1A(entt::entity source, entt::entity target, float sigma2, float c);

    void P1PX(entt::entity source, entt::entity target, float sigma2, float c);


}

#endif //ENGINE24_SURFACEMESHCOMPUTE_H
