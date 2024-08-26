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

        explicit GraphInterface(GraphData &data) : GraphInterface(data.vertices, data.halfedges, data.edges) {}

        GraphInterface(Vertices &vertices,
                       HalfEdges &halfEdges,
                       Edges &edges) :
                vertices(vertices),
                halfedges(halfEdges),
                edges(edges),
                vpoint(vertices.get_vertex_property<PointType>("v:point")),
                vconnectivity(vertices.get_vertex_property<Halfedge>("v:connectivity")),
                ecolors(edges.get_edge_property<ColorType>("e:color")),
                escalarfield(edges.get_edge_property<ScalarType>("e:scalarfield")),
                hconnectivity(halfEdges.get_halfedge_property<HalfedgeConnectivity>("h:connectivity")) {}

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

        template<class T>
        inline HalfedgeProperty<T> add_halfedge_property(const std::string &name,
                                                         const T t = T()) {
            return HalfedgeProperty<T>(halfedges.add<T>(name, t));
        }

        template<class T>
        inline HalfedgeProperty<T> get_halfedge_property(const std::string &name) const {
            return HalfedgeProperty<T>(halfedges.get<T>(name));
        }

        template<class T>
        inline HalfedgeProperty<T> halfedge_property(const std::string &name, const T t = T()) {
            return HalfedgeProperty<T>(halfedges.get_or_add<T>(name, t));
        }

        template<class T>
        inline void remove_halfedge_property(HalfedgeProperty<T> &p) {
            halfedges.remove(p);
        }

        inline bool has_halfedge_property(const std::string &name) const {
            return halfedges.exists(name);
        }

        template<class T>
        inline EdgeProperty<T> add_edge_property(const std::string &name,
                                                 const T t = T()) {
            return EdgeProperty<T>(edges.add<T>(name, t));
        }

        template<class T>
        inline EdgeProperty<T> get_edge_property(const std::string &name) const {
            return EdgeProperty<T>(edges.get<T>(name));
        }

        template<class T>
        inline EdgeProperty<T> edge_property(const std::string &name, const T t = T()) {
            return EdgeProperty<T>(edges.get_or_add<T>(name, t));
        }

        template<class T>
        inline void remove_edge_property(EdgeProperty<T> &p) {
            edges.remove(p);
        }

        inline bool has_edge_property(const std::string &name) const {
            return edges.exists(name);
        }

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

    struct GraphOwning : public GraphInterface {
        GraphOwning() : GraphInterface(data) {}

    private:
        GraphData data;
    };
}

#endif //ENGINE24_GRAPHINTERFACE_H
