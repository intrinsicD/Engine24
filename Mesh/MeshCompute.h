//
// Created by alex on 21.06.24.
//

#ifndef ENGINE24_MESHCOMPUTE_H
#define ENGINE24_MESHCOMPUTE_H

#include "PluginMesh.h"

namespace Bcg {
    unsigned int CompileComputeShader(const char *source);

    VertexProperty<Vector<float, 3>> ComputeVertexNormals(SurfaceMesh &mesh);

    FaceProperty<Vector<float, 3>> ComputeFaceNormals(SurfaceMesh &mesh);
}

#endif //ENGINE24_MESHCOMPUTE_H
