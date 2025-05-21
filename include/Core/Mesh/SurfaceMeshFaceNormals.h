//
// Created by alex on 11/26/24.
//

#ifndef SURFACEMESHFACENORMALS_H
#define SURFACEMESHFACENORMALS_H

#include "SurfaceMesh.h"

namespace Bcg {
    Vector<float, 3> compute_normal(const SurfaceMesh &mesh, Face f);

    FaceProperty<Vector<float, 3> > compute_face_normals(SurfaceMesh &mesh);
}


#endif //SURFACEMESHFACENORMALS_H
