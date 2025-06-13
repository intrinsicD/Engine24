//
// Created by alex on 13.06.25.
//

#ifndef ENGINE24_MESH_H
#define ENGINE24_MESH_H

#include <vector>
#include "MatVec.h"

namespace Bcg{
    struct Mesh {
        std::vector<Vector<float, 3>> positions; // Vertex positions
        std::vector<Vector<float, 3>> normals; // Vertex normals
        std::vector<Vector<float, 3>> colors; // Vertex colors
        std::vector<Vector<float, 2>> uvs; // Texture coordinates
        std::vector<Vector<unsigned int, 3>> faces; // Triangle indices
    };
}

#endif //ENGINE24_MESH_H
