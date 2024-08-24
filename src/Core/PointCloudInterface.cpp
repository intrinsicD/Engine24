//
// Created by alex on 24.08.24.
//

#include "PointCloudInterface.h"
#include "Logger.h"

namespace Bcg {
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

    void PointCloudInterface::set_scalarfield(const std::vector<ScalarType> &scalarfield) {
        if (scalarfield.size() != vertices.size()) {
            Log::Error("Number of scalarfield does not match number of vertices");
            return;
        }
        if (!vcolor) {
            vcolor = vertices.add_vertex_property<ColorType>("v:scalarfield");
        }
        vscalarfield.vector() = scalarfield;
    }

    Vertex PointCloudInterface::new_vertex() {
        if (vertices.size() == BCG_MAX_INDEX - 1) {
            auto what =
                    "SurfaceMesh: cannot allocate vertex, max. index reached";
            throw AllocationException(what);
        }
        vertices.push_back();
        return Vertex(static_cast<IndexType>(vertices.size()) - 1);
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
            vpoint[v] = p;
        return v;
    }

    void PointCloudInterface::garbage_collection() {
        if (!vertices.has_garbage_)
            return;

        auto nV = vertices.size();

        // setup handle mapping
        VertexProperty<Vertex> vmap = vertices.add_vertex_property<Vertex>("v:garbage-collection");

        for (size_t i = 0; i < nV; ++i)
            vmap[Vertex(i)] = Vertex(i);

        // remove deleted vertices
        if (nV > 0) {
            size_t i0 = 0;
            size_t i1 = nV - 1;

            while (true) {
                // find first deleted and last un-deleted
                while (!vertices.vdeleted[Vertex(i0)] && i0 < i1)
                    ++i0;
                while (vertices.vdeleted[Vertex(i1)] && i0 < i1)
                    --i1;
                if (i0 >= i1)
                    break;

                // swap
                vertices.swap(i0, i1);
            }

            // remember new size
            nV = vertices.vdeleted[Vertex(i0)] ? i0 : i0 + 1;
        }

        // remove handle maps
        vertices.remove_vertex_property(vmap);

        // finally resize arrays
        vertices.resize(nV);
        vertices.free_memory();

        vertices.deleted_vertices = 0;
        vertices.has_garbage_ = false;
    }
}