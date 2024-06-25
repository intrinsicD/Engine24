//
// Created by alex on 21.06.24.
//

#ifndef ENGINE24_MESHCOMPUTE_H
#define ENGINE24_MESHCOMPUTE_H

#include "PluginMesh.h"

namespace Bcg {
    unsigned int CompileComputeShader(const char *source);

    pmp::VertexProperty<pmp::Vector<float, 3>> ComputeVertexNormals(pmp::SurfaceMesh &mesh);
    pmp::FaceProperty<pmp::Vector<float, 3>> ComputeFaceNormals(pmp::SurfaceMesh &mesh);
}

#endif //ENGINE24_MESHCOMPUTE_H
