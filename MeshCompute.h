//
// Created by alex on 21.06.24.
//

#ifndef ENGINE24_MESHCOMPUTE_H
#define ENGINE24_MESHCOMPUTE_H

#include "PluginMesh.h"

namespace Bcg {
    unsigned int CompileComputeShader(const char *source);

    std::vector<float> ComputeFaceNormals(MeshComponent &mesh);
}

#endif //ENGINE24_MESHCOMPUTE_H
