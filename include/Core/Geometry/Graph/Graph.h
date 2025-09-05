//
// Created by alex on 26.08.24.
//

#ifndef ENGINE24_GRAPH_H
#define ENGINE24_GRAPH_H


#include "Properties.h"
#include "Types.h"
#include "GeometryCommon.h"
#include "GeometryData.h"

namespace Bcg {
    class Graph {
    public:
        using VertexIterator = Iterator<Graph, Vertex>;
        using VertexContainer = HandleContainer<Graph, Vertex>;
        using HalfedgeIterator = Iterator<Graph, Halfedge>;
        using HalfedgeContainer = HandleContainer<Graph, Halfedge>;
        using EdgeIterator = Iterator<Graph, Edge>;
        using EdgeContainer = HandleContainer<Graph, Edge>;

        Graph();

        //! destructor
        virtual ~Graph() = default;

        //! copy constructor: copies \p rhs to \p *this. performs a deep copy of all
        //! properties.
        inline Graph(const Graph &rhs) { operator=(rhs); }

        //! assign \p rhs to \p *this. performs a deep copy of all properties.
        Graph &operator=(const Graph &rhs);

        //! assign \p rhs to \p *this. does not copy custom properties.
        Graph &assign(const Graph &rhs);

        //! add a new vertex with position \p p
        Vertex add_vertex(const PointType &p);

        void mark_vertex_deleted(Vertex v);

        void remove_vertex(Vertex v);

        //! \return number of (deleted and valid) vertices in the mesh
        inline size_t vertices_size() const { return vprops_.size(); }

        inline size_t halfedges_size() const { return hprops_.size(); }

        inline size_t edges_size() const { return eprops_.size(); }

        //! \return number of vertices in the mesh
        inline size_t n_vertices() const { return vertices_size() - deleted_vertices_; }

        inline size_t n_halfedges() const { return halfedges_size() - deleted_halfedges_; }

        inline size_t n_edges() const { return edges_size() - deleted_edges_; }

        //! \return true if the mesh is empty, i.e., has no vertices
        inline bool is_empty() const { return n_vertices() == 0; }

        //! clear mesh: remove all vertices, edges, faces
        virtual void clear();

        //! remove unused memory from vectors
        void free_memory();

        //! reserve memory (mainly used in file readers)
        void reserve(size_t nvertices, size_t nedges);

        //! remove deleted elements
        void garbage_collection();

        //! \return whether vertex \p v is deleted
        //! \sa garbage_collection()
        inline bool is_deleted(Vertex v) const { return vdeleted_[v]; }

        //! \return whether vertex \p v is valid.
        inline bool is_valid(Vertex v) const { return (v.idx() < vertices_size()); }

        inline bool is_deleted(Halfedge h) const { return hdeleted_[h]; }

        inline bool is_valid(Halfedge h) const { return (h.idx() < halfedges_size()); }

        inline bool is_deleted(Edge e) const { return edeleted_[e]; }

        inline bool is_valid(Edge e) const { return (e.idx() < edges_size()); }

        //!@}
        //! \name Property handling
        //!@{

        //! add a vertex property of type \p T with name \p name and default
        //! value \p t. fails if a property named \p name exists already,
        //! since the name has to be unique. in this case it returns an
        //! invalid property
        template<class T>
        inline VertexProperty<T> add_vertex_property(const std::string &name,
                                                     const T t = T()) {
            return VertexProperty<T>(vprops_.add<T>(name, t));
        }

        //! get the vertex property named \p name of type \p T. returns an
        //! invalid VertexProperty if the property does not exist or if the
        //! type does not match.
        template<class T>
        inline VertexProperty<T> get_vertex_property(const std::string &name) const {
            return VertexProperty<T>(vprops_.get<T>(name));
        }

        //! if a vertex property of type \p T with name \p name exists, it is
        //! returned. otherwise this property is added (with default value \c
        //! t)
        template<class T>
        inline VertexProperty<T> vertex_property(const std::string &name, const T t = T()) {
            return VertexProperty<T>(vprops_.get_or_add<T>(name, t));
        }

        //! remove the vertex property \p p
        template<class T>
        inline void remove_vertex_property(VertexProperty<T> &p) {
            vprops_.remove(p);
        }

        //! does the mesh have a vertex property with name \p name?
        inline bool has_vertex_property(const std::string &name) const {
            return vprops_.exists(name);
        }

        //! add a halfedge property of type \p T with name \p name and default
        //! value \p t.  fails if a property named \p name exists already,
        //! since the name has to be unique. in this case it returns an
        //! invalid property.
        template<class T>
        inline HalfedgeProperty<T> add_halfedge_property(const std::string &name,
                                                         const T t = T()) {
            return HalfedgeProperty<T>(hprops_.add<T>(name, t));
        }

        //! add a edge property of type \p T with name \p name and default
        //! value \p t.  fails if a property named \p name exists already,
        //! since the name has to be unique.  in this case it returns an
        //! invalid property.
        template<class T>
        inline EdgeProperty<T> add_edge_property(const std::string &name, const T t = T()) {
            return EdgeProperty<T>(eprops_.add<T>(name, t));
        }

        //! get the halfedge property named \p name of type \p T. returns an
        //! invalid VertexProperty if the property does not exist or if the
        //! type does not match.
        template<class T>
        inline HalfedgeProperty<T> get_halfedge_property(const std::string &name) const {
            return HalfedgeProperty<T>(hprops_.get<T>(name));
        }

        //! get the edge property named \p name of type \p T. returns an
        //! invalid VertexProperty if the property does not exist or if the
        //! type does not match.
        template<class T>
        inline EdgeProperty<T> get_edge_property(const std::string &name) const {
            return EdgeProperty<T>(eprops_.get<T>(name));
        }

        //! if a halfedge property of type \p T with name \p name exists, it is
        //! returned.  otherwise this property is added (with default value \c
        //! t)
        template<class T>
        inline HalfedgeProperty<T> halfedge_property(const std::string &name,
                                                     const T t = T()) {
            return HalfedgeProperty<T>(hprops_.get_or_add<T>(name, t));
        }

        //! if an edge property of type \p T with name \p name exists, it is
        //! returned.  otherwise this property is added (with default value \c
        //! t)
        template<class T>
        inline EdgeProperty<T> edge_property(const std::string &name, const T t = T()) {
            return EdgeProperty<T>(eprops_.get_or_add<T>(name, t));
        }

        //! remove the halfedge property \p p
        template<class T>
        inline void remove_halfedge_property(HalfedgeProperty<T> &p) {
            hprops_.remove(p);
        }

        //! does the mesh have a halfedge property with name \p name?
        inline bool has_halfedge_property(const std::string &name) const {
            return hprops_.exists(name);
        }

        //! remove the edge property \p p
        template<class T>
        inline void remove_edge_property(EdgeProperty<T> &p) {
            eprops_.remove(p);
        }

        //! does the mesh have an edge property with name \p name?
        inline bool has_edge_property(const std::string &name) const {
            return eprops_.exists(name);
        }

        //! \return the names of all vertex properties
        inline std::vector<std::string> vertex_properties() const {
            return vprops_.properties();
        }

        //!@}
        //! \name Iterators and circulators
        //!@{

        //! \return start iterator for vertices
        VertexIterator vertices_begin() const {
            return VertexIterator(Vertex(0), this);
        }

        //! \return end iterator for vertices
        VertexIterator vertices_end() const {
            return VertexIterator(Vertex(static_cast<IndexType>(vertices_size())),
                                  this);
        }

        HalfedgeIterator halfedges_begin() const {
            return HalfedgeIterator(Halfedge(0), this);
        }

        HalfedgeIterator halfedges_end() const {
            return HalfedgeIterator(Halfedge(static_cast<IndexType>(halfedges_size())),
                                    this);
        }

        EdgeIterator edges_begin() const {
            return EdgeIterator(Edge(0), this);
        }

        EdgeIterator edges_end() const {
            return EdgeIterator(Edge(static_cast<IndexType>(edges_size())),
                                this);
        }

        //! \return vertex container for C++11 range-based for-loops
        VertexContainer vertices() const {
            return VertexContainer(vertices_begin(), vertices_end());
        }

        HalfedgeContainer halfedges() const {
            return HalfedgeContainer(halfedges_begin(), halfedges_end());
        }

        EdgeContainer edges() const {
            return EdgeContainer(edges_begin(), edges_end());
        }

        //! position of a vertex (read only)
        inline const PointType &position(Vertex v) const { return vpoint_[v]; }

        //! position of a vertex
        inline PointType &position(Vertex v) { return vpoint_[v]; }

        //! \return vector of point positions
        inline std::vector<PointType> &positions() { return vpoint_.vector(); }

        inline Vertex new_vertex() {
            if (vertices_size() == BCG_MAX_INDEX - 1) {
                auto what =
                        "SurfaceMesh: cannot allocate vertex, max. index reached";
                throw AllocationException(what);
            }
            vprops_.push_back();
            return Vertex(static_cast<IndexType>(vertices_size()) - 1);
        }

        // are there any deleted entities?
        inline bool has_garbage() const { return has_garbage_; }


        Property<Vector<IndexType, 2> > get_edges();

        inline bool is_isolated(Vertex v) const {
            return is_valid(get_halfedge(v)) && is_valid(get_opposite(get_halfedge(v)));
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

        Halfedge find_halfedge(Vertex v0, Vertex v1) const;

        inline Halfedge get_opposite(Halfedge h) const {
            return Halfedge((h.idx() & 1) ? h.idx() - 1 : h.idx() + 1);
        }

        inline void set_halfedge(Vertex v0, Halfedge h) {
            vconn_[v0] = h;
        }

        inline Halfedge get_halfedge(Vertex v0) const {
            return vconn_[v0];
        }

        inline Halfedge get_halfedge(Edge e, int i) const {
            return Halfedge{(e.idx() << 1) + i};
        }

        inline Vertex get_vertex(Edge e, int i) const {
            return to_vertex(get_halfedge(e, i));
        }

        inline void set_vertex(Halfedge h, Vertex v) {
            hconn_[h].v = v;
        }

        inline Vertex to_vertex(Halfedge h) const {
            return hconn_[h].v;
        }

        inline Vertex from_vertex(Halfedge h) const {
            return to_vertex(get_opposite(h));
        }

        inline Halfedge get_next(Halfedge h) const {
            return hconn_[h].nh;
        }

        inline void set_next(Halfedge h, Halfedge nh) {
            hconn_[h].nh = nh;
            hconn_[nh].ph = h;
        }

        inline Halfedge get_prev(Halfedge h) const {
            return hconn_[h].ph;
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

        Halfedge new_edge(Vertex v0, Vertex v1);

        Halfedge add_edge(Vertex v0, Vertex v1);

        void mark_edge_deleted(Edge e);

        void remove_edge(Edge e);

        void mark_halfedge_deleted(Halfedge h);

        size_t get_valence(Vertex v) const;

        inline VertexAroundVertexCirculatorBase<Graph> get_vertices(Vertex v) const {
            return {this, v};
        }

        inline HalfedgeAroundVertexCirculatorBase<Graph> get_halfedges(Vertex v) const {
            return {this, v};
        }

        inline EdgeAroundVertexCirculatorBase<Graph> get_edges(Vertex v) const {
            return {this, v};
        }

        Vertices vprops_;
        Halfedges hprops_;
        Edges eprops_;

        // point coordinates
        VertexProperty<PointType> vpoint_;
        VertexProperty<Halfedge> vconn_;

        HalfedgeProperty<HalfedgeConnectivity> hconn_;

        // markers for deleted entities
        VertexProperty<bool> vdeleted_;
        HalfedgeProperty<bool> hdeleted_;
        EdgeProperty<bool> edeleted_;

        // numbers of deleted entities
        IndexType deleted_vertices_{0};
        IndexType deleted_halfedges_{0};
        IndexType deleted_edges_{0};

        bool has_garbage_{false};
    };
}

#endif //ENGINE24_GRAPH_H
