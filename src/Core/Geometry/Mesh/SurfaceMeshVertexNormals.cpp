//
// Created by alex on 11/26/24.
//

#include "SurfaceMeshVertexNormals.h"
#include "Eigen/Geometry"

namespace Bcg {
    Eigen::Vector<float, 3> compute_normal(const SurfaceMesh &mesh, Vertex v) {
        Eigen::Vector<float, 3> v_normal = Eigen::Vector<float, 3>::Zero();
        const Eigen::Vector<float, 3> &v0 = mesh.position(v);

        for(auto h : mesh.halfedges(v)) {
            auto v1 = mesh.to_vertex(h);
            auto v2 = mesh.to_vertex(mesh.next_halfedge(h));
            Eigen::Vector<float, 3> f_normal = (mesh.position(v1) - v0).cross(mesh.position(v2) - v0);
            float area = f_normal.norm() / 2.0f;
            v_normal += f_normal * area;
        }
        return v_normal.normalized();
    }

    VertexProperty<Eigen::Vector<float, 3> > compute_vertex_normals(SurfaceMesh &mesh) {
        VertexProperty<Eigen::Vector<float, 3> > normals = mesh.vertex_property<Eigen::Vector<float, 3> >("normals");
        for (auto v: mesh.vertices()) {
            normals[v] = compute_normal(mesh, v);
        }
        return normals;
    }
}
