// Created by alex on 24.08.24.
//

#ifndef ENGINE24_POINTCLOUDINTERFACE_H
#define ENGINE24_POINTCLOUDINTERFACE_H

#include "GeometryData.h"

namespace Bcg {
    struct PointCloudInterface {
        explicit PointCloudInterface(PointCloudData &data) : PointCloudInterface(data.vertices) {
        }

        explicit PointCloudInterface(Vertices &vertices) : vertices(vertices),
                                                           vpoint(vertices.vertex_property<PointType>("v:point")),
                                                           vnormal(vertices.vertex_property<NormalType>("v:normal")),
                                                           vcolor(vertices.vertex_property<ColorType>("v:color")),
                                                           vscalarfield(
                                                               vertices.vertex_property<ScalarType>("v:scalarfield")),
                                                           vradius(vertices.vertex_property<ScalarType>("v:radius")),
                                                           vdeleted(vertices.vertex_property<bool>("v:deleted")) {
        }

        PointCloudInterface(PointCloudInterface &&other) noexcept
            : vertices(other.vertices) {
            vpoint = other.vpoint;
            vnormal = other.vnormal;
            vcolor = other.vcolor;
            vscalarfield = other.vscalarfield;
            vradius = other.vradius;
            vdeleted = other.vdeleted;
        }

        PointCloudInterface &operator=(PointCloudInterface &&other) noexcept {
            if (this != &other) {
                vertices = other.vertices;
                vpoint = other.vpoint;
                vnormal = other.vnormal;
                vcolor = other.vcolor;
                vscalarfield = other.vscalarfield;
                vradius = other.vradius;
                vdeleted = other.vdeleted;
            }
            return *this;
        }

        Vertices &vertices;

        VertexProperty<PointType> vpoint;
        VertexProperty<NormalType> vnormal;
        VertexProperty<ColorType> vcolor;
        VertexProperty<ScalarType> vscalarfield;
        VertexProperty<ScalarType> vradius;
        VertexProperty<bool> vdeleted;

        template<class T>
        VertexProperty<T> add_vertex_property(const std::string &name, const T t = T()) {
            return VertexProperty<T>(vertices.add<T>(name, t));
        }

        template<class T>
        VertexProperty<T> get_vertex_property(const std::string &name) const {
            return VertexProperty<T>(vertices.get<T>(name));
        }

        template<class T>
        VertexProperty<T> vertex_property(const std::string &name, const T t = T()) {
            return VertexProperty<T>(vertices.get_or_add<T>(name, t));
        }

        template<class T>
        void remove_vertex_property(VertexProperty<T> &p) {
            vertices.remove(p);
        }

        [[nodiscard]] bool has_vertex_property(const std::string &name) const {
            return vertices.exists(name);
        }

        [[nodiscard]] bool is_empty() const { return vertices.size() == 0; }

        [[nodiscard]] bool is_deleted(Vertex v) const { return vdeleted[v]; }

        [[nodiscard]] bool is_valid(Vertex v) const { return (v.idx() < vertices.size()); }

        void set_points(const std::vector<PointType> &points);

        void set_normals(const std::vector<NormalType> &normals);

        void set_colors(const std::vector<ColorType> &colors);

        void set_scalarfield(const std::vector<ScalarType> &scalarfield);

        void set_radii(const std::vector<ScalarType> &radii);

        Vertex new_vertex();

        Vertex add_vertex(const PointType &p);

        void mark_vertex_deleted(Vertex v);

        void remove_vertex(Vertex v);

        void garbage_collection();
    };
}

#endif //ENGINE24_POINTCLOUDINTERFACE_H
