//
// Created by alex on 28.06.24.
//

#ifndef ENGINE24_UTILS_H
#define ENGINE24_UTILS_H

#include "SurfaceMesh.h"

namespace Bcg{
    FaceProperty<Vector<unsigned int, 3>> SurfaceMeshTriangles(SurfaceMesh &mesh);
}

#endif //ENGINE24_UTILS_H
