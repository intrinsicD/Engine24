//
// Created by alex on 21.06.24.
//

#ifndef ENGINE24_MESHCOMPUTE_H
#define ENGINE24_MESHCOMPUTE_H

#include "PluginMesh.h"

namespace Bcg {
    VertexProperty<Vector<float, 3>> ComputeVertexNormals(entt::entity entity_id, SurfaceMesh &mesh);

    FaceProperty<Vector<float, 3>> ComputeFaceNormals(entt::entity entity_id, SurfaceMesh &mesh);

    void PT1A(entt::entity source, entt::entity target, float sigma2, float c);

    void P1PX(entt::entity source, entt::entity target, float sigma2, float c);


}

#endif //ENGINE24_MESHCOMPUTE_H
