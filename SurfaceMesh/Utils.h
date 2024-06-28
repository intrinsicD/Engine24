//
// Created by alex on 28.06.24.
//

#ifndef ENGINE24_UTILS_H
#define ENGINE24_UTILS_H

#include "SurfaceMesh.h"

namespace Bcg{
    FaceProperty<Vector<unsigned int, 3>> extract_triangle_list(SurfaceMesh &mesh);
}

#endif //ENGINE24_UTILS_H
