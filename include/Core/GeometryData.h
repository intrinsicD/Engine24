//
// Created by alex on 22.08.24.
//

#ifndef ENGINE24_GEOMETRYDATA_H
#define ENGINE24_GEOMETRYDATA_H

#include "Properties.h"

namespace Bcg {
    struct Vertices : public PropertyContainer {
    };

    struct HalfEdges : public PropertyContainer {
    };

    struct Edges : public PropertyContainer {
    };

    struct Faces : public PropertyContainer {
    };

    struct PointCloudData {
        Vertices vertices;
    };

    struct GraphData {
        Vertices vertices;
        HalfEdges halfedges;
        Edges edges;
    };

    struct MeshData {
        Vertices vertices;
        HalfEdges halfedges;
        Edges edges;
        Faces faces;
    };

    struct PointCloudInterface {
        PointCloudInterface(PointCloudData &data) : vertices(data.vertices) {}

        PointCloudInterface(Vertices &vertices) : vertices(vertices) {}

        Vertices &vertices;

        VertexProperty <PointType> vpoint;
        VertexProperty <NormalType> vnormal;
        VertexProperty <ColorType> vcolor;
        VertexProperty <ScalarType> vradius;

        void set_points(const std::vector<PointType> &points) {
            if (points.size() != vertices.size()) {
                Log::Error("Number of points does not match number of vertices");
                return;
            }
            if (!vpoint) {
                vpoint = vertices.add_vertex_property<PointType>("v:point");
            }
            vpoint.vector() = points;
        }

        void set_normals(const std::vector<NormalType> &normals) {
            if (normals.size() != vertices.size()) {
                Log::Error("Number of normals does not match number of vertices");
                return;
            }
            if (!vnormal) {
                vnormal = vertices.add_vertex_property<NormalType>("v:normal");
            }
            vnormal.vector() = normals;
        }

        void set_colors(const std::vector<ColorType> &colors) {
            if (colors.size() != vertices.size()) {
                Log::Error("Number of colors does not match number of vertices");
                return;
            }
            if (!vcolor) {
                vcolor = vertices.add_vertex_property<ColorType>("v:color");
            }
            vcolor.vector() = colors;
        }

        void set_radii(const std::vector<ScalarType> &radii) {
            if (radii.size() != vertices.size()) {
                Log::Error("Number of radii does not match number of vertices");
                return;
            }
            if (!vradius) {
                vradius = vertices.add_vertex_property<ScalarType>("v:radius");
            }
            vradius.vector() = radii;
        }

        Vertex add_vertex(const PointType &p) {
            Vertex v = new_vertex();
            if (v.is_valid())
                vpoint_[v] = p;
            return v;
        }
    };
}

#endif //ENGINE24_GEOMETRYDATA_H
