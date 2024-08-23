//
// Created by alex on 22.08.24.
//

#ifndef ENGINE24_GEOMETRYDATA_H
#define ENGINE24_GEOMETRYDATA_H

#include "Properties.h"
#include "Types.h"
#include "GeometryCommon.h"

namespace Bcg {
    struct Vertices : public PropertyContainer {
        Vertices() : deleted_vertices(0) {}

        VertexProperty<bool> vdeleted;
        size_t deleted_vertices;

        inline bool is_deleted(Vertex v) const { return vdeleted[v]; }

        //! \return whether vertex \p v is valid.
        inline bool is_valid(Vertex v) const { return (v.idx() < size()); }

        template<class T>
        inline VertexProperty<T> add_vertex_property(const std::string &name,
                                                     const T t = T()) {
            return VertexProperty<T>(add<T>(name, t));
        }

        //! get the vertex property named \p name of type \p T. returns an
        //! invalid VertexProperty if the property does not exist or if the
        //! type does not match.
        template<class T>
        inline VertexProperty<T> get_vertex_property(const std::string &name) const {
            return VertexProperty<T>(get<T>(name));
        }

        //! if a vertex property of type \p T with name \p name exists, it is
        //! returned. otherwise this property is added (with default value \c
        //! t)
        template<class T>
        inline VertexProperty<T> vertex_property(const std::string &name, const T t = T()) {
            return VertexProperty<T>(get_or_add<T>(name, t));
        }

        //! remove the vertex property \p p
        template<class T>
        inline void remove_vertex_property(VertexProperty<T> &p) {
            remove(p);
        }

        //! does the mesh have a vertex property with name \p name?
        inline bool has_vertex_property(const std::string &name) const {
            return exists(name);
        }
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


        void set_points(const std::vector<PointType> &points);

        void set_normals(const std::vector<NormalType> &normals);

        void set_colors(const std::vector<ColorType> &colors);

        void set_radii(const std::vector<ScalarType> &radii);

        Vertex add_vertex(const PointType &p);
    };

    void PointCloudInterface::set_points(const std::vector<PointType> &points) {
        if (points.size() != vertices.size()) {
            Log::Error("Number of points does not match number of vertices");
            return;
        }
        if (!vpoint) {
            vpoint = vertices.add_vertex_property<PointType>("v:point");
        }
        vpoint.vector() = points;
    }

    void PointCloudInterface::set_normals(const std::vector<NormalType> &normals) {
        if (normals.size() != vertices.size()) {
            Log::Error("Number of normals does not match number of vertices");
            return;
        }
        if (!vnormal) {
            vnormal = vertices.add_vertex_property<NormalType>("v:normal");
        }
        vnormal.vector() = normals;
    }

    void PointCloudInterface::set_colors(const std::vector<ColorType> &colors) {
        if (colors.size() != vertices.size()) {
            Log::Error("Number of colors does not match number of vertices");
            return;
        }
        if (!vcolor) {
            vcolor = vertices.add_vertex_property<ColorType>("v:color");
        }
        vcolor.vector() = colors;
    }

    void PointCloudInterface::set_radii(const std::vector<ScalarType> &radii) {
        if (radii.size() != vertices.size()) {
            Log::Error("Number of radii does not match number of vertices");
            return;
        }
        if (!vradius) {
            vradius = vertices.add_vertex_property<ScalarType>("v:radius");
        }
        vradius.vector() = radii;
    }

    Vertex PointCloudInterface::add_vertex(const PointType &p) {
        Vertex v = new_vertex();
        if (v.is_valid())
            vpoint_[v] = p;
        return v;
    }

    struct PointCloud : public PointCloudInterface {
        PointCloud() : PointCloudInterface(vertices_) {}

        Vertices vertices_;
    };
}

#endif //ENGINE24_GEOMETRYDATA_H
