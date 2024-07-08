//
// Created by alex on 08.07.24.
//

#include "Mesh.h"

namespace Bcg{
    FaceProperty<Vector<unsigned int, 3>> extract_triangle_list(SurfaceMesh &mesh) {
        auto triangles = mesh.get_face_property<Vector<unsigned int, 3>>("f:indices");
        if (!triangles) {
            triangles = mesh.add_face_property<Vector<unsigned int, 3>>("f:indices");
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

    void
    extract_triangle_list(SurfaceMesh &mesh, std::vector<float> &vertices, std::vector<unsigned int> &indices) {
        vertices.clear();
        indices.clear();

        auto vertexIndexMap = mesh.add_vertex_property<unsigned int>("v:index_map");

        unsigned int index = 0;

        // Extract vertex positions
        for (auto v: mesh.vertices()) {
            Point p = mesh.position(v);
            vertices.push_back(p[0]);
            vertices.push_back(p[1]);
            vertices.push_back(p[2]);
            vertexIndexMap[v] = index++;
        }

        // Extract triangle indices
        for (auto f: mesh.faces()) {
            std::vector<unsigned int> faceIndices;
            for (auto v: mesh.vertices(f)) {
                faceIndices.push_back(vertexIndexMap[v]);
            }
            if (faceIndices.size() == 3) {
                indices.push_back(faceIndices[0]);
                indices.push_back(faceIndices[1]);
                indices.push_back(faceIndices[2]);
            } else {
                Log::Error("Warning: Non-triangular face encountered. Ignoring face.");
            }
        }

        mesh.remove_vertex_property(vertexIndexMap);
    }
}