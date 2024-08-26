//
// Created by alex on 24.08.24.
//

#ifndef ENGINE24_POINTCLOUDINTERFACE_H
#define ENGINE24_POINTCLOUDINTERFACE_H

#include "GeometryData.h"

namespace Bcg {
    struct PointCloudInterface {
        explicit PointCloudInterface(PointCloudData &data) : PointCloudInterface(data.vertices) {}

        explicit PointCloudInterface(Vertices &vertices) :
                vertices(vertices),
                vpoint(vertices.get_vertex_property<PointType>("v:point")),
                vnormal(vertices.get_vertex_property<NormalType>("v:normal")),
                vcolor(vertices.get_vertex_property<ColorType>("v:color")),
                vscalarfield(vertices.get_vertex_property<ScalarType>("v:scalarfield")),
                vradius(vertices.get_vertex_property<ScalarType>("v:radius")) {}

        Vertices &vertices;

        VertexProperty<PointType> vpoint;
        VertexProperty<NormalType> vnormal;
        VertexProperty<ColorType> vcolor;
        VertexProperty<ScalarType> vscalarfield;
        VertexProperty<ScalarType> vradius;

        template<class T>
        inline VertexProperty<T> add_vertex_property(const std::string &name,
                                                     const T t = T()) {
            return VertexProperty<T>(vertices.add<T>(name, t));
        }

        template<class T>
        inline VertexProperty<T> get_vertex_property(const std::string &name) const {
            return VertexProperty<T>(vertices.get<T>(name));
        }

        template<class T>
        inline VertexProperty<T> vertex_property(const std::string &name, const T t = T()) {
            return VertexProperty<T>(vertices.get_or_add<T>(name, t));
        }

        template<class T>
        inline void remove_vertex_property(VertexProperty<T> &p) {
            vertices.remove(p);
        }

        inline bool has_vertex_property(const std::string &name) const {
            return vertices.exists(name);
        }

        void set_points(const std::vector<PointType> &points);

        void set_normals(const std::vector<NormalType> &normals);

        void set_colors(const std::vector<ColorType> &colors);

        void set_scalarfield(const std::vector<ScalarType> &scalarfield);

        void set_radii(const std::vector<ScalarType> &radii);

        Vertex new_vertex();

        Vertex add_vertex(const PointType &p);

        void garbage_collection();
    };

    struct PointCloudOwning : public PointCloudInterface {
        PointCloudOwning() : PointCloudInterface(vertices_) {}

    private:
        Vertices vertices_;
    };
}

#endif //ENGINE24_POINTCLOUDINTERFACE_H
