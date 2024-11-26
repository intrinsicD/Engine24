//
// Created by alex on 11/26/24.
//

#include "SurfaceMeshVertexNormals.h"

namespace Bcg {
    Vector<float, 3> compute_normal(const SurfaceMesh &mesh, Vertex v) {
        Vector<float, 3> v_normal(0);
        const Vector<float, 3> &v0 = mesh.position(v);

        for(auto h : mesh.halfedges(v)) {
            auto v1 = mesh.to_vertex(h);
            auto v2 = mesh.to_vertex(mesh.next_halfedge(h));
            Vector<float, 3> f_normal = glm::cross(mesh.position(v1) - v0, mesh.position(v2) - v0);
            float area = glm::length(f_normal) / 2.0f;
            v_normal += f_normal * area;
        }
        return glm::normalize(v_normal);
    }

    VertexProperty<Vector<float, 3> > compute_vertex_normals(SurfaceMesh &mesh) {
        VertexProperty<Vector<float, 3> > normals = mesh.vertex_property<Vector<float, 3> >("normals");
        for (auto v: mesh.vertices()) {
            normals[v] = compute_normal(mesh, v);
        }
        return normals;
    }
}
