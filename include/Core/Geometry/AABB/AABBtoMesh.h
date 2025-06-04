//
// Created by alex on 04.06.25.
//

#ifndef ENGINE24_AABBTOMESH_H
#define ENGINE24_AABBTOMESH_H

#include "AABB.h"
#include "SurfaceMesh.h"

namespace Bcg{
    template<typename T>
    SurfaceMesh BuildMesh(const AABB<T> &aabb) {
        SurfaceMesh mesh;
        auto v0 = mesh.add_vertex(aabb.min);
        auto v1 = mesh.add_vertex(Vector<T, 3>(aabb.min.x, aabb.min.y, aabb.max.z));
        auto v2 = mesh.add_vertex(Vector<T, 3>(aabb.min.x, aabb.max.y, aabb.min.z));
        auto v3 = mesh.add_vertex(Vector<T, 3>(aabb.min.x, aabb.max.y, aabb.max.z));
        auto v4 = mesh.add_vertex(Vector<T, 3>(aabb.max.x, aabb.min.y, aabb.min.z));
        auto v5 = mesh.add_vertex(Vector<T, 3>(aabb.max.x, aabb.min.y, aabb.max.z));
        auto v6 = mesh.add_vertex(Vector<T, 3>(aabb.max.x, aabb.max.y, aabb.min.z));
        auto v7 = mesh.add_vertex(aabb.max);

        mesh.add_quad(v0, v1, v3, v2); // Bottom
        mesh.add_quad(v4, v5, v7, v6); // Top
        mesh.add_quad(v0, v2, v6, v4); // Left
        mesh.add_quad(v1, v3, v7, v5); // Right
        mesh.add_quad(v0, v1, v5, v4); // Front
        mesh.add_quad(v2, v3, v7, v6); // Back

        return mesh;
    }
}

#endif //ENGINE24_AABBTOMESH_H
