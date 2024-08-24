//
// Created by alex on 24.08.24.
//

#include "GraphInterface.h"
#include "Logger.h"

namespace Bcg {
    void GraphInterface::set_points(const std::vector<PointType> &points) {
        if (points.size() != vertices.size()) {
            Log::Error("Number of points does not match number of vertices");
            return;
        }
        if (!vpoint) {
            vpoint = vertices.add_vertex_property<PointType>("v:point");
        }
        vpoint.vector() = points;
    }

    void GraphInterface::set_edge_colors(const std::vector<ColorType> &colors) {
        if (colors.size() != edges.size()) {
            Log::Error("Number of edge colors does not match number of edges");
            return;
        }
        if (!ecolors) {
            ecolors = edges.add_edge_property<PointType>("e:color");
        }
        ecolors.vector() = colors;
    }

    void GraphInterface::set_edge_scalarfield(const std::vector<ScalarType> &scalarfield) {
        if (scalarfield.size() != edges.size()) {
            Log::Error("Number of edge scalarfield does not match number of edges");
            return;
        }
        if (!escalarfield) {
            escalarfield = edges.add_edge_property<ScalarType>("e:scalarfield");
        }
        escalarfield.vector() = scalarfield;
    }

    Property<Vector<IndexType, 2>> GraphInterface::get_edges() const {
        auto indices = edges.edge_property<Vector<IndexType, 2>>("e:indices");
        for (auto e: edges) {
            indices[e] = {get_vertex(get_halfedge(e, 0)).idx(), get_vertex(get_halfedge(e, 1)).idx()};
        }
        return indices;
    }

    Vertex GraphInterface::new_vertex() {
        if (vertices.size() == BCG_MAX_INDEX - 1) {
            auto what = "GraphInterface: cannot allocate vertex, max. index reached";
            throw AllocationException(what);
        }
        vertices.push_back();
        return Vertex(static_cast<IndexType>(vertices.size()) - 1);
    }

    Vertex GraphInterface::add_vertex(const PointType &p) {
        auto v = new_vertex();
        vpoint[v] = p;
        return v;
    }

    Halfedge GraphInterface::new_edge(Vertex v0, Vertex v1) {
        if (v0 == v1) {
            Log::Error("GraphInterface: cannot create edge with same vertices");
            return Halfedge();
        }

        edges.push_back();
        halfedges.push_back();
        halfedges.push_back();

        Halfedge h(halfedges.size() - 2);
        Halfedge o(halfedges.size() - 1);

        set_vertex(h, v1);
        set_vertex(o, v0);
        return h;
    }

    Halfedge GraphInterface::add_edge(Vertex v0, Vertex v1) {
        Halfedge h01 = find_halfedge(v0, v1);
        if (halfedges.is_valid(h01)) {
            return h01;
        }

        Halfedge out_h0 = get_halfedge(v0);
        Halfedge out_h1 = get_halfedge(v1);

        Halfedge h = new_edge(v0, v1);
        Halfedge o = get_opposite(h);

        if (halfedges.is_valid(out_h1)) {
            Halfedge nh = get_next(out_h1);
            Halfedge ph = get_prev(out_h1);

            set_next(h, nh);
            set_prev(nh, h);

            set_next(ph, o);
            set_prev(o, ph);
        } else {
            set_next(h, o);
            set_prev(o, h);
        }

        set_halfedge(v1, o);

        if (halfedges.is_valid(out_h0)) {
            Halfedge nh = get_next(out_h0);
            Halfedge ph = get_prev(out_h0);

            set_next(o, nh);
            set_prev(nh, o);

            set_next(ph, h);
            set_prev(h, ph);
        } else {
            set_next(o, h);
            set_prev(h, o);
        }

        set_halfedge(v0, h);

        return h;
    }

    Halfedge GraphInterface::find_halfedge(Vertex v0, Vertex v1) const {
        if (!halfedges.is_valid(get_halfedge(v0))) {
            return Halfedge();
        }
        for (auto h: halfedges) {
            if (get_vertex(h) == v1) {
                return h;
            }
        }
        return {};
    }

    size_t GraphInterface::get_valence(Vertex v) const {
        size_t valence = 0;
        for (auto h: halfedges) {
            if (get_vertex(h) == v) {
                ++valence;
            }
        }
        return valence;
    }

    void GraphInterface::remove_edge(Edge e) {
        if (edges.edeleted[e]) {
            return;
        }

        edges.edeleted[e] = true;


        Halfedge h = get_halfedge(e, 0);
        Halfedge o = get_halfedge(e, 1);

        halfedges.hdeleted[h] = true;
        halfedges.hdeleted[o] = true;

        Vertex v0 = get_vertex(h);
        Vertex v1 = get_vertex(o);

        Halfedge out_v1 = get_next(h);
        Halfedge in_v1 = get_prev(o);

        if (out_v1 != o) {
            set_prev(out_v1, in_v1);
            set_next(in_v1, out_v1);
        } else {
            set_halfedge(v1, Halfedge());
        }

        Halfedge out_v0 = get_next(o);
        Halfedge in_v0 = get_prev(h);

        if (out_v0 != h) {
            set_prev(out_v0, in_v0);
            set_next(in_v0, out_v0);
        } else {
            set_halfedge(v0, Halfedge());
        }

        ++edges.deleted_edges;
        ++halfedges.deleted_halfedges;
        ++halfedges.deleted_halfedges;
    }

    void GraphInterface::garbage_collection() {}

    Vertex GraphInterface::split(Edge e, Vertex v) {
        Halfedge h = get_halfedge(e, 0);
        Halfedge o = get_halfedge(e, 1);

        Halfedge nh = get_next(h);
        Halfedge po = get_prev(o);

        Vertex v1 = get_vertex(h);

        Halfedge new_h = add_edge(v, v1);
        Halfedge new_o = get_opposite(new_h);

        set_next(new_h, nh);
        set_prev(nh, new_h);
        set_prev(new_h, h);
        set_next(h, new_h);

        set_prev(new_o, po);
        set_next(po, new_o);
        set_next(new_o, o);
        set_prev(o, new_o);

        set_halfedge(v, new_h);

        return v;
    }

    Vertex GraphInterface::split(Edge e, PointType point) {
        return split(e, add_vertex(point));
    }

    Vertex GraphInterface::collapse(Edge e, ScalarType t) {}
}