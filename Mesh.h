//
// Created by alex on 25.06.24.
//

#ifndef ENGINE24_MESH_H
#define ENGINE24_MESH_H

#include "pmp/surface_mesh.h"

namespace Bcg {
    using Mesh = pmp::SurfaceMesh;

    pmp::FaceProperty<pmp::Vector<unsigned int, 3>> extract_triangle_list(pmp::SurfaceMesh &mesh) {
        auto triangles = mesh.get_face_property<pmp::Vector<unsigned int, 3>>("f:indices");
        if (!triangles) {
            triangles = mesh.add_face_property<pmp::Vector<unsigned int, 3>>("f:indices");
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
    extract_triangle_list(pmp::SurfaceMesh &mesh, std::vector<float> &vertices, std::vector<unsigned int> &indices) {
        vertices.clear();
        indices.clear();

        auto vertexIndexMap = mesh.add_vertex_property<unsigned int>("v:index_map");

        unsigned int index = 0;

        // Extract vertex positions
        for (auto v: mesh.vertices()) {
            pmp::Point p = mesh.position(v);
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
                std::cerr << "Warning: Non-triangular face encountered. Ignoring face." << std::endl;
            }
        }

        mesh.remove_vertex_property(vertexIndexMap);
    }
}

#endif //ENGINE24_MESH_H
