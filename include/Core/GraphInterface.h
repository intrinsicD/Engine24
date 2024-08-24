//
// Created by alex on 24.08.24.
//

#ifndef ENGINE24_GRAPHINTERFACE_H
#define ENGINE24_GRAPHINTERFACE_H

#include "GeometryData.h"

namespace Bcg {
    struct GraphInterface {
        GraphInterface(GraphData &data) : vertices(data.vertices), halfedges(data.halfedges), edges(data.edges) {}

        GraphInterface(Vertices &vertices, HalfEdges &halfEdges, Edges &edges) : vertices(vertices),
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
        };
        HalfedgeProperty<HalfedgeConnectivity> hconnectivity;

        void set_points(const std::vector<PointType> &points);

        void set_edge_colors(const std::vector<ColorType> &colors);

        void set_edge_scalarfield(const std::vector<ScalarType> &escalarfield);

        Vertex new_vertex();

        Vertex add_vertex(const PointType &p);

        Halfedge new_edge(Vertex v0, Vertex v1);

        Halfedge add_edge(Vertex v0, Vertex v1);

        Halfedge find_halfedge(Vertex v0, Vertex v1) const;

        Halfedge get_opposite(Halfedge h) const;

        Halfedge get_halfedge(Vertex v0) const;

        Halfedge get_halfedge(Edge e, int i) const;



        void set_vertex(Halfedge h, Vertex v);

        void set_halfedge(Vertex v, Halfedge h);

        Vertex get_vertex(Halfedge h) const;

        void set_next(Halfedge h, Halfedge nh);

        Halfedge get_next(Halfedge h) const;

        void set_prev(Halfedge h, Halfedge ph);

        Halfedge get_prev(Halfedge h) const;

        void remove_edge(Edge e);

        void garbage_collection();

        Vertex split(Edge e, ScalarType t = 0.5); //t ranges from 0 to 1

        Vertex collapse(Edge e, ScalarType t = 0.5); //t ranges from 0 to 1
    };
}

#endif //ENGINE24_GRAPHINTERFACE_H
