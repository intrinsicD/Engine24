//
// Created by alex on 25.08.24.
//

#ifndef ENGINE24_MESHINTERFACE_H
#define ENGINE24_MESHINTERFACE_H

#include "GeometryData.h"

namespace Bcg {
    struct HalfedgeMeshInterface {
        using VertexAroundVertexCirculator = VertexAroundVertexCirculatorBase<HalfedgeMeshInterface>;
        using HalfedgeAroundVertexCirculator = HalfedgeAroundVertexCirculatorBase<HalfedgeMeshInterface>;
        using EdgeAroundVertexCirculator = EdgeAroundVertexCirculatorBase<HalfedgeMeshInterface>;
        using FaceAroundVertexCirculator = FaceAroundVertexCirculatorBase<HalfedgeMeshInterface>;
        using VertexAroundFaceCirculator = VertexAroundFaceCirculatorBase<HalfedgeMeshInterface>;
        using HalfedgeAroundFaceCirculator = HalfedgeAroundFaceCirculatorBase<HalfedgeMeshInterface>;

        explicit HalfedgeMeshInterface(MeshData &data) : HalfedgeMeshInterface(data.vertices, data.halfedges,
                                                                               data.edges, data.faces) {}

        HalfedgeMeshInterface(Vertices &vertices,
                              HalfEdges &halfEdges,
                              Edges &edges, Faces &faces) :
                vertices(vertices),
                halfedges(halfEdges),
                edges(edges),
                faces(faces),
                vpoint(vertices.get_vertex_property<PointType>("v:point")),
                vconnectivity(vertices.get_vertex_property<Halfedge>("v:connectivity")),
                hconnectivity(halfEdges.get_halfedge_property<HalfedgeConnectivity>("h:connectivity")),
                fconnectivity(faces.get_face_property<Halfedge>("f:connectivity")),
                fcolors(faces.get_face_property<ColorType>("f:color")),
                fscalarfield(faces.get_face_property<ScalarType>("f:scalarfield")) {}

        Vertices &vertices;
        HalfEdges &halfedges;
        Edges &edges;
        Faces &faces;

        VertexProperty<PointType> vpoint;
        VertexProperty<Halfedge> vconnectivity;

        struct HalfedgeConnectivity {
            Face f;
            Vertex v;
            Halfedge nh, ph;

            friend std::ostream &operator<<(std::ostream &os, const HalfedgeConnectivity &hc) {
                os << "f: " << hc.f.idx_
                   << "v: " << hc.v.idx_
                   << "nh: " << hc.nh.idx_
                   << "ph: " << hc.ph.idx_;
                return os;
            }
        };

        HalfedgeProperty<HalfedgeConnectivity> hconnectivity;

        FaceProperty<Halfedge> fconnectivity;
        FaceProperty<ColorType> fcolors;
        FaceProperty<ScalarType> fscalarfield;

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

        template<class T>
        inline FaceProperty<T> add_face_property(const std::string &name,
                                                 const T t = T()) {
            return FaceProperty<T>(faces.add<T>(name, t));
        }

        template<class T>
        inline FaceProperty<T> get_face_property(const std::string &name) const {
            return FaceProperty<T>(faces.get<T>(name));
        }

        template<class T>
        inline FaceProperty<T> face_property(const std::string &name, const T t = T()) {
            return FaceProperty<T>(faces.get_or_add<T>(name, t));
        }

        template<class T>
        inline void remove_face_property(FaceProperty<T> &p) {
            faces.remove(p);
        }

        inline bool has_face_property(const std::string &name) const {
            return faces.exists(name);
        }

        void set_points(const std::vector<PointType> &points);

        void set_face_colors(const std::vector<ColorType> &colors);

        void set_face_scalarfield(const std::vector<ScalarType> &fscalarfield);

        FaceProperty<Vector<IndexType, 3>> get_triangles() const;

        void reserve(size_t n_vertices, size_t n_edges, size_t n_faces);

        void resize(size_t n_vertices, size_t n_edges, size_t n_faces);

        Vertex add_vertex(const PointType &p);

        Face add_face(const std::vector<Vertex> &vertices);

        Face add_triangle(Vertex v0, Vertex v1, Vertex v2);

        Face add_quad(Vertex v0, Vertex v1, Vertex v2, Vertex v3);

        void garbage_collection();

        inline Halfedge get_halfedge(Vertex v) const { return vconnectivity[v]; }

        inline void set_halfedge(Vertex v, Halfedge h) { vconnectivity[v] = h; }

        inline bool is_boundary(Vertex v) const {
            Halfedge h(get_halfedge(v));
            return (!(h.is_valid() && face(h).is_valid()));
        }

        inline bool is_isolated(Vertex v) const { return !get_halfedge(v).is_valid(); }

        bool is_manifold(Vertex v) const;

        inline Vertex to_vertex(Halfedge h) const { return hconnectivity[h].v; }

        inline Vertex from_vertex(Halfedge h) const {
            return to_vertex(opposite_halfedge(h));
        }

        inline void set_vertex(Halfedge h, Vertex v) { hconnectivity[h].v = v; }

        inline Face face(Halfedge h) const { return hconnectivity[h].f; }

        inline void set_face(Halfedge h, Face f) { hconnectivity[h].f = f; }

        inline Halfedge next_halfedge(Halfedge h) const {
            return hconnectivity[h].nh;
        }

        inline void set_next_halfedge(Halfedge h, Halfedge nh) {
            hconnectivity[h].nh = nh;
            hconnectivity[nh].ph = h;
        }

        inline void set_prev_halfedge(Halfedge h, Halfedge ph) {
            hconnectivity[h].ph = ph;
            hconnectivity[ph].nh = h;
        }

        inline Halfedge prev_halfedge(Halfedge h) const {
            return hconnectivity[h].ph;
        }

        inline Halfedge opposite_halfedge(Halfedge h) const {
            return Halfedge((h.idx() & 1) ? h.idx() - 1 : h.idx() + 1);
        }

        inline Halfedge rotate_ccw(Halfedge h) const {
            return opposite_halfedge(prev_halfedge(h));
        }

        inline Halfedge rotate_cw(Halfedge h) const {
            return next_halfedge(opposite_halfedge(h));
        }

        inline Edge edge(Halfedge h) const { return Edge(h.idx() >> 1); }

        inline bool is_boundary(Halfedge h) const { return !face(h).is_valid(); }

        inline Halfedge get_halfedge(Edge e, unsigned int i) const {
            assert(i <= 1);
            return Halfedge((e.idx() << 1) + i);
        }

        inline Vertex vertex(Edge e, unsigned int i) const {
            assert(i <= 1);
            return to_vertex(get_halfedge(e, i));
        }

        inline Face face(Edge e, unsigned int i) const {
            assert(i <= 1);
            return face(get_halfedge(e, i));
        }

        inline bool is_boundary(Edge e) const {
            return (is_boundary(get_halfedge(e, 0)) || is_boundary(get_halfedge(e, 1)));
        }

        inline Halfedge get_halfedge(Face f) const { return fconnectivity[f]; }

        inline void set_halfedge(Face f, Halfedge h) { fconnectivity[f] = h; }

        bool is_boundary(Face f) const;

        inline VertexAroundVertexCirculator get_vertices(Vertex v) const {
            return {this, v};
        }

        inline EdgeAroundVertexCirculator get_edges(Vertex v) const {
            return {this, v};
        }

        inline HalfedgeAroundVertexCirculator get_halfedges(Vertex v) const {
            return {this, v};
        }

        inline FaceAroundVertexCirculator get_faces(Vertex v) const {
            return {this, v};
        }

        inline VertexAroundFaceCirculator get_vertices(Face f) const {
            return {this, f};
        }

        inline HalfedgeAroundFaceCirculator get_halfedges(Face f) const {
            return {this, f};
        }

        inline Halfedge insert_vertex(Edge e, const PointType &p) {
            return insert_vertex(get_halfedge(e, 0), add_vertex(p));
        }

        inline Halfedge insert_vertex(Edge e, Vertex v) {
            return insert_vertex(get_halfedge(e, 0), v);
        }

        Halfedge insert_vertex(Halfedge h0, Vertex v);

        Halfedge find_halfedge(Vertex start, Vertex end) const;

        Edge find_edge(Vertex a, Vertex b) const;

        bool is_triangle_mesh() const;

        bool is_quad_mesh() const;

        bool is_collapse_ok(Halfedge v0v1) const;

        void collapse(Halfedge h);

        bool is_removal_ok(Edge e) const;

        bool remove_edge(Edge e);

        inline Vertex split(Face f, const PointType &p) {
            Vertex v = add_vertex(p);
            split(f, v);
            return v;
        }

        void split(Face f, Vertex v);

        inline Halfedge split(Edge e, const PointType &p) { return split(e, add_vertex(p)); }

        Halfedge split(Edge e, Vertex v);

        Halfedge insert_edge(Halfedge h0, Halfedge h1);

        bool is_flip_ok(Edge e) const;

        void flip(Edge e);

        size_t valence(Vertex v) const;

        size_t valence(Face f) const;

        void delete_vertex(Vertex v);

        void delete_edge(Edge e);

        void delete_face(Face f);

        inline Vertex new_vertex() {
            vertices.push_back();
            return Vertex(static_cast<IndexType>(vertices.size()) - 1);
        }

        Halfedge new_edge();

        Halfedge new_edge(Vertex start, Vertex end);

        inline Face new_face() {
            faces.push_back();
            return Face(static_cast<IndexType>(faces.size()) - 1);
        }

        void adjust_outgoing_halfedge(Vertex v);

        void remove_edge_helper(Halfedge h);

        void remove_loop_helper(Halfedge h);

    private:
        // helper data for add_face()
        using NextCacheEntry = std::pair<Halfedge, Halfedge>;
        using NextCache = std::vector<NextCacheEntry>;
        std::vector<Vertex> add_face_vertices_;
        std::vector<Halfedge> add_face_halfedges_;
        std::vector<bool> add_face_is_new_;
        std::vector<bool> add_face_needs_adjust_;
        NextCache add_face_next_cache_;
    };

    struct MeshOwning : public HalfedgeMeshInterface {
        MeshOwning() : HalfedgeMeshInterface(data) {}

    private:
        MeshData data;
    };
}

#endif //ENGINE24_MESHINTERFACE_H
