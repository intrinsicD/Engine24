//
// Created by alex on 12.08.24.
//

#include "SurfaceMesh.h"

namespace Bcg {
    FaceProperty<Vector<unsigned int, 3>> SurfaceMeshTriangles(SurfaceMesh &mesh) {
        auto triangles = mesh.get_face_property<Vector<unsigned int, 3 >>("f:indices");
        if (!triangles) {
            triangles = mesh.add_face_property<Vector<unsigned int, 3 >>("f:indices");
        }
        for (auto f: mesh.faces()) {
            std::vector<unsigned int> faceIndices;
            for (auto v: mesh.vertices(f)) {
                faceIndices.push_back(v.idx());
            }
            if (faceIndices.size() == 3) {
                triangles[f][0] = faceIndices[0];
                triangles[f][1] = faceIndices[1];
                triangles[f][2] = faceIndices[2];
            }
        }

        return triangles;
    }
}