//
// Created by alex on 24.08.24.
//

#ifndef ENGINE24_GRAPHINTERFACE_H
#define ENGINE24_GRAPHINTERFACE_H

#include "GeometryData.h"

namespace Bcg {
    struct GraphInterface {
        using VertexAroundVertexCirculator = VertexAroundVertexCirculatorBase<GraphInterface>;
        using HalfedgeAroundVertexCirculator = HalfedgeAroundVertexCirculatorBase<GraphInterface>;

        explicit GraphInterface(GraphData &data) : vertices(data.vertices),
                                                   halfedges(data.halfedges),
                                                   edges(data.edges) {}

        GraphInterface(Vertices &vertices,
                       HalfEdges &halfEdges,
                       Edges &edges) : vertices(vertices),
                                       halfedges(halfEdges),
                                       edges(edges) {}

        Vertices &vertices;
        HalfEdges &halfedges;
        Edges &edges;

        VertexProperty<PointType> vpoint;
        VertexProperty<Halfedge> vconnectivity;

        EdgeProperty<ColorType> ecolors;
        EdgeProperty<ScalarType> escalarfield;

        struct HalfedgeConnectivity {
            Vertex v;
            Halfedge nh;
            Halfedge ph;

            friend std::ostream &operator<<(std::ostream &os, const HalfedgeConnectivity &hc) {
                os << "v: " << hc.v.idx_
                   << "nh: " << hc.nh.idx_
                   << "ph: " << hc.ph.idx_;
                return os;
            }
        };

        HalfedgeProperty<HalfedgeConnectivity> hconnectivity;

        void set_points(const std::vector<PointType> &points);

        void set_edge_colors(const std::vector<ColorType> &colors);

        void set_edge_scalarfield(const std::vector<ScalarType> &escalarfield);

        Property<Vector<IndexType, 2>> get_edges() const;

        inline bool is_isolated(Vertex v) const {
            return halfedges.is_valid(get_halfedge(v)) && halfedges.is_valid(get_opposite(get_halfedge(v)));
        }

        inline bool is_boundary(Vertex v) const {
            return is_boundary(get_halfedge(v));
        }

        inline bool is_boundary(Halfedge h) const {
            return get_next(h) == get_opposite(h);
        }

        inline bool is_boundary(Edge e) const {
            return is_boundary(get_halfedge(e, 0)) || is_boundary(get_halfedge(e, 1));
        }

        Vertex new_vertex();

        Vertex add_vertex(const PointType &p);

        Halfedge new_edge(Vertex v0, Vertex v1);

        Halfedge add_edge(Vertex v0, Vertex v1);

        Halfedge find_halfedge(Vertex v0, Vertex v1) const;

        inline Halfedge get_opposite(Halfedge h) const {
            return Halfedge((h.idx() & 1) ? h.idx() - 1 : h.idx() + 1);
        }

        inline Halfedge get_halfedge(Vertex v0) const {
            return vconnectivity[v0];
        }

        inline Halfedge get_halfedge(Edge e, int i) const {
            return Halfedge{(e.idx() << 1) + i};
        }

        inline Vertex get_vertex(Edge e, int i) const {
            return get_vertex(get_halfedge(e, i));
        }

        inline void set_vertex(Halfedge h, Vertex v) {
            hconnectivity[h].v = v;
        }

        inline void set_halfedge(Vertex v, Halfedge h) {
            vconnectivity[v] = h;
        }

        inline Vertex get_vertex(Halfedge h) const {
            return hconnectivity[h].v;
        }

        inline void set_next(Halfedge h, Halfedge nh) {
            hconnectivity[h].nh = nh;
        }

        inline Halfedge get_next(Halfedge h) const {
            return hconnectivity[h].nh;
        }

        inline void set_prev(Halfedge h, Halfedge ph) {
            hconnectivity[h].ph = ph;
        }

        inline Halfedge get_prev(Halfedge h) const {
            return hconnectivity[h].ph;
        }

        inline Halfedge rotate_cw(Halfedge h) const {
            return get_next(get_opposite(h));
        }

        inline Halfedge rotate_ccw(Halfedge h) const {
            return get_opposite(get_prev(h));
        }

        inline Edge get_edge(Halfedge h) const {
            return Edge(h.idx() >> 1);
        }

        size_t get_valence(Vertex v) const;

        void remove_edge(Edge e);

        void garbage_collection();

        Vertex split(Edge e, Vertex v);

        Vertex split(Edge e, PointType point);

        Vertex split(Edge e, ScalarType t = 0.5);

        void collapse(Edge e, ScalarType t = 0.5); //t ranges from 0 to 1

        inline VertexAroundVertexCirculator get_vertices(Vertex v) const {
            return {this, v};
        }

        inline HalfedgeAroundVertexCirculator get_halfedges(Vertex v) const {
            return {this, v};
        }


    };
}

#endif //ENGINE24_GRAPHINTERFACE_H
