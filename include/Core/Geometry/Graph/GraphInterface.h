//
// Created by alex on 24.08.24.
//

#ifndef ENGINE24_GRAPHINTERFACE_H
#define ENGINE24_GRAPHINTERFACE_H

#include "GeometryData.h"

#include <utility>

namespace Bcg {
    struct GraphInterface {
        using VertexAroundVertexCirculator = VertexAroundVertexCirculatorBase<GraphInterface>;
        using HalfedgeAroundVertexCirculator = HalfedgeAroundVertexCirculatorBase<GraphInterface>;

        explicit GraphInterface(GraphData &data) : GraphInterface(data.vertices, data.halfedges, data.edges) {
        }

        GraphInterface(Vertices &vertices_,
                       Halfedges &halfedges_,
                       Edges &edges_) : vertices(vertices_),
                                        halfedges(halfedges_),
                                        edges(edges_) {
            vpoint = vertices.vertex_property<PointType>("v:point");
            vconnectivity = vertices.vertex_property<Halfedge>("v:connectivity");
            vdeleted = vertices.vertex_property<bool>("v:deleted");
            ecolors = edges.edge_property<ColorType>("e:color");
            hconnectivity = halfedges.halfedge_property<HalfedgeConnectivity>("h:connectivity");
            escalarfield = edges.edge_property<ScalarType>("e:scalarfield");
        }

        // Define move constructor
        GraphInterface(GraphInterface &&other) noexcept
            : vertices(other.vertices),
              halfedges(other.halfedges),
              edges(other.edges) {
            vpoint = other.vpoint;
            vconnectivity = other.vconnectivity;
            vdeleted = other.vdeleted;
            ecolors = other.ecolors;
            hconnectivity = other.hconnectivity;
            escalarfield = other.escalarfield;
        }

        // Define move assignment operator
        GraphInterface &operator=(GraphInterface &&other) noexcept {
            if (this != &other) {
                vertices = std::move(other.vertices);
                halfedges = std::move(other.halfedges);
                edges = std::move(other.edges);
                vpoint = other.vpoint;
                vconnectivity = other.vconnectivity;
                ecolors = other.ecolors;
                hconnectivity = other.hconnectivity;
                escalarfield = other.escalarfield;
            }
            return *this;
        }

        Vertices &vertices;
        Halfedges &halfedges;
        Edges &edges;

        VertexProperty<PointType> vpoint;
        VertexProperty<Halfedge> vconnectivity;
        VertexProperty<bool> vdeleted;

        EdgeProperty<ColorType> ecolors;
        EdgeProperty<ScalarType> escalarfield;

        HalfedgeProperty<HalfedgeConnectivity> hconnectivity;

        template<class T>
        VertexProperty<T> add_vertex_property(const std::string &name,
                                              const T t = T()) {
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

        template<class T>
        HalfedgeProperty<T> add_halfedge_property(const std::string &name,
                                                  const T t = T()) {
            return HalfedgeProperty<T>(halfedges.add<T>(name, t));
        }

        template<class T>
        HalfedgeProperty<T> get_halfedge_property(const std::string &name) const {
            return HalfedgeProperty<T>(halfedges.get<T>(name));
        }

        template<class T>
        HalfedgeProperty<T> halfedge_property(const std::string &name, const T t = T()) {
            return HalfedgeProperty<T>(halfedges.get_or_add<T>(name, t));
        }

        template<class T>
        void remove_halfedge_property(HalfedgeProperty<T> &p) {
            halfedges.remove(p);
        }

        [[nodiscard]] bool has_halfedge_property(const std::string &name) const {
            return halfedges.exists(name);
        }

        template<class T>
        EdgeProperty<T> add_edge_property(const std::string &name,
                                          const T t = T()) {
            return EdgeProperty<T>(edges.add<T>(name, t));
        }

        template<class T>
        EdgeProperty<T> get_edge_property(const std::string &name) const {
            return EdgeProperty<T>(edges.get<T>(name));
        }

        template<class T>
        EdgeProperty<T> edge_property(const std::string &name, const T t = T()) {
            return EdgeProperty<T>(edges.get_or_add<T>(name, t));
        }

        template<class T>
        void remove_edge_property(EdgeProperty<T> &p) {
            edges.remove(p);
        }

        [[nodiscard]] bool has_edge_property(const std::string &name) const {
            return edges.exists(name);
        }

        void set_points(const std::vector<PointType> &points);

        void set_edge_colors(const std::vector<ColorType> &colors);

        void set_edge_scalarfield(const std::vector<ScalarType> &escalarfield);

        [[nodiscard]] Property<Vector<IndexType, 2> > get_edges() const;

        [[nodiscard]] bool is_valid(Vertex v) const {
            return v.idx() < vertices.size();
        }

        [[nodiscard]] bool is_valid(Halfedge h) const {
            return h.idx() < halfedges.size();
        }

        [[nodiscard]] bool is_valid(Edge e) const {
            return e.idx() < edges.size();
        }

        [[nodiscard]] bool is_isolated(Vertex v) const {
            return is_valid(get_halfedge(v)) && is_valid(get_opposite(get_halfedge(v)));
        }

        [[nodiscard]] bool is_boundary(Vertex v) const {
            return is_boundary(get_halfedge(v));
        }

        [[nodiscard]] bool is_boundary(Halfedge h) const {
            return get_next(h) == get_opposite(h);
        }

        [[nodiscard]] bool is_boundary(Edge e) const {
            return is_boundary(get_halfedge(e, 0)) || is_boundary(get_halfedge(e, 1));
        }

        Vertex new_vertex();

        Vertex add_vertex(const PointType &p);

        Halfedge new_edge(Vertex v0, Vertex v1);

        Halfedge add_edge(Vertex v0, Vertex v1);

        [[nodiscard]] Halfedge find_halfedge(Vertex v0, Vertex v1) const;

        [[nodiscard]] Halfedge get_opposite(Halfedge h) const {
            return Halfedge{(h.idx() & 1) ? h.idx() - 1 : h.idx() + 1};
        }

        [[nodiscard]] Halfedge get_halfedge(Vertex v0) const {
            return vconnectivity[v0];
        }

        [[nodiscard]] Halfedge get_halfedge(Edge e, int i) const {
            return Halfedge{(e.idx() << 1) + i};
        }

        [[nodiscard]] Vertex get_vertex(Edge e, int i) const {
            return to_vertex(get_halfedge(e, i));
        }

        void set_vertex(Halfedge h, Vertex v) {
            hconnectivity[h].v = v;
        }

        void set_halfedge(Vertex v, Halfedge h) {
            vconnectivity[v] = h;
        }

        [[nodiscard]] Vertex from_vertex(Halfedge h) const {
            return to_vertex(get_opposite(h));
        }

        [[nodiscard]] Vertex to_vertex(Halfedge h) const {
            return hconnectivity[h].v;
        }

        void set_next(Halfedge h, Halfedge nh) {
            hconnectivity[h].nh = nh;
            hconnectivity[nh].ph = h;
        }

        [[nodiscard]] Halfedge get_next(Halfedge h) const {
            return hconnectivity[h].nh;
        }

        [[nodiscard]] Halfedge get_prev(Halfedge h) const {
            return hconnectivity[h].ph;
        }

        [[nodiscard]] Halfedge rotate_cw(Halfedge h) const {
            return get_next(get_opposite(h));
        }

        [[nodiscard]] Halfedge rotate_ccw(Halfedge h) const {
            return get_opposite(get_prev(h));
        }

        [[nodiscard]] Edge get_edge(Halfedge h) const {
            return Edge{h.idx() >> 1};
        }

        [[nodiscard]] size_t get_valence(Vertex v) const;

        void remove_edge(Edge e);

        void garbage_collection();

        void clear();

        void reserve(size_t nvertices, size_t nedges);

        void free_memory();

        Vertex split(Edge e, Vertex v);

        Vertex split(Edge e, PointType point);

        Vertex split(Edge e, ScalarType t = 0.5);

        void collapse(Edge e, ScalarType t = 0.5); //t ranges from 0 to 1

        [[nodiscard]] VertexAroundVertexCirculator get_vertices(Vertex v) const {
            return {this, v};
        }

        [[nodiscard]] HalfedgeAroundVertexCirculator get_halfedges(Vertex v) const {
            return {this, v};
        }
    };
}

#endif //ENGINE24_GRAPHINTERFACE_H
