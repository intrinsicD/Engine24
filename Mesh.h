//
// Created by alex on 25.06.24.
//

#ifndef ENGINE24_MESH_H
#define ENGINE24_MESH_H

#include "SurfaceMesh/SurfaceMesh.h"
#include "Core/Logger.h"

namespace Bcg {
    using Mesh = SurfaceMesh;

    FaceProperty<Vector<unsigned int, 3>> extract_triangle_list(SurfaceMesh &mesh);

    void extract_triangle_list(SurfaceMesh &mesh, std::vector<float> &vertices,
                               std::vector<unsigned int> &indices);
}

#endif //ENGINE24_MESH_H
