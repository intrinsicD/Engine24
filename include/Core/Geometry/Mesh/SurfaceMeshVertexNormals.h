//
// Created by alex on 11/26/24.
//

#ifndef SURFACEMESHVERTEXNORMALS_H
#define SURFACEMESHVERTEXNORMALS_H

#include "SurfaceMesh.h"

namespace Bcg {
    Vector<float, 3> compute_normal(const SurfaceMesh &mesh, Vertex v);

    VertexProperty<Vector<float, 3>> compute_vertex_normals(SurfaceMesh &mesh);
}

#endif //SURFACEMESHVERTEXNORMALS_H
