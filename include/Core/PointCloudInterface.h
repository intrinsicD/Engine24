//
// Created by alex on 24.08.24.
//

#ifndef ENGINE24_POINTCLOUDINTERFACE_H
#define ENGINE24_POINTCLOUDINTERFACE_H

#include "GeometryData.h"

namespace Bcg {
    struct PointCloudInterface {
        explicit PointCloudInterface(PointCloudData &data) : vertices(data.vertices) {}

        explicit PointCloudInterface(Vertices &vertices) : vertices(vertices) {}

        Vertices &vertices;

        VertexProperty<PointType> vpoint;
        VertexProperty<NormalType> vnormal;
        VertexProperty<ColorType> vcolor;
        VertexProperty<ScalarType> vscalarfield;
        VertexProperty<ScalarType> vradius;

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
