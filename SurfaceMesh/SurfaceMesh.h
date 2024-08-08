// Copyright 2011-2021 the Polygon Mesh Processing Library developers.
// Copyright 2001-2005 by Computer Graphics Group, RWTH Aachen
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <cassert>
#include <cstddef>
#include <compare>
#include <filesystem>
#include <iterator>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "Types.h"
#include "Properties.h"
#include "GeometryCommon.h"

namespace Bcg {

    struct IOFlags;

//! \brief A class for representing polygon surface meshes.
//! \details This class implements a half-edge data structure for surface meshes.
//! See \cite sieger_2011_design for details on the design and implementation.
//! \note This class only supports 2-manifold surface meshes with boundary.
    class SurfaceMesh {
    public:
        //! \name Iterator Types
        //!@{

        using VertexIterator = Iterator<SurfaceMesh, Vertex>;
        using HalfedgeIterator = Iterator<SurfaceMesh, Halfedge>;
        using EdgeIterator = Iterator<SurfaceMesh, Edge>;
        using FaceIterator = Iterator<SurfaceMesh, Face>;

        //!@}
        //! \name Container Types
        //!@{

        //! helper class for iterating through all vertices using range-based
        //! for-loops.
        using VertexContainer = HandleContainer<SurfaceMesh, Vertex>;
        using HalfedgeContainer = HandleContainer<SurfaceMesh, Halfedge>;
        using EdgeContainer = HandleContainer<SurfaceMesh, Edge>;
        using FaceContainer = HandleContainer<SurfaceMesh, Face>;

        //!@}
        //! \name Circulator Types
        //!@{

        //! this class circulates through all one-ring neighbors of a vertex.
        //! it also acts as a container-concept for C++11 range-based for loops.
        //! \sa HalfedgeAroundVertexCirculator, vertices(Vertex)
        using VertexAroundVertexCirculator = VertexAroundVertexCirculatorBase<class SurfaceMesh>;
        using HalfedgeAroundVertexCirculator = HalfedgeAroundVertexCirculatorBase<class SurfaceMesh>;
        using EdgeAroundVertexCirculator = EdgeAroundVertexCirculatorBase<class SurfaceMesh>;
        using FaceAroundVertexCirculator = FaceAroundVertexCirculatorBase<class SurfaceMesh>;
        using VertexAroundFaceCirculator = VertexAroundFaceCirculatorBase<class SurfaceMesh>;
        using HalfedgeAroundFaceCirculator = HalfedgeAroundFaceCirculatorBase<class SurfaceMesh>;

        //!@}
        //! \name Construction, destruction, assignment
        //!@{

        //! default constructor
        SurfaceMesh();

        //! destructor
        virtual ~SurfaceMesh();

        //! copy constructor: copies \p rhs to \p *this. performs a deep copy of all
        //! properties.
        inline SurfaceMesh(const SurfaceMesh &rhs) { operator=(rhs); }

        //! assign \p rhs to \p *this. performs a deep copy of all properties.
        SurfaceMesh &operator=(const SurfaceMesh &rhs);

        //! assign \p rhs to \p *this. does not copy custom properties.
        SurfaceMesh &assign(const SurfaceMesh &rhs);

        //!@}
        //! \name Add new elements by hand
        //!@{

        //! add a new vertex with position \p p
        Vertex add_vertex(const PointType &p);

        //! \brief Add a new face with vertex list \p vertices
        //! \throw TopologyException in case a topological error occurs.
        //! \sa add_triangle, add_quad
        Face add_face(const std::vector<Vertex> &vertices);

        //! add a new triangle connecting vertices \p v0, \p v1, \p v2
        //! \sa add_face, add_quad
        Face add_triangle(Vertex v0, Vertex v1, Vertex v2);

        //! add a new quad connecting vertices \p v0, \p v1, \p v2, \p v3
        //! \sa add_triangle, add_face
        Face add_quad(Vertex v0, Vertex v1, Vertex v2, Vertex v3);

        //!@}
        //! \name Memory Management
        //!@{

        //! \return number of (deleted and valid) vertices in the mesh
        inline size_t vertices_size() const { return vprops_.size(); }

        //! \return number of (deleted and valid) halfedges in the mesh
        inline size_t halfedges_size() const { return hprops_.size(); }

        //! \return number of (deleted and valid) edges in the mesh
        inline size_t edges_size() const { return eprops_.size(); }

        //! \return number of (deleted and valid) faces in the mesh
        inline size_t faces_size() const { return fprops_.size(); }

        //! \return number of vertices in the mesh
        inline size_t n_vertices() const { return vertices_size() - deleted_vertices_; }

        //! \return number of halfedge in the mesh
        inline size_t n_halfedges() const { return halfedges_size() - 2 * deleted_edges_; }

        //! \return number of edges in the mesh
        inline size_t n_edges() const { return edges_size() - deleted_edges_; }

        //! \return number of faces in the mesh
        inline size_t n_faces() const { return faces_size() - deleted_faces_; }

        //! \return true if the mesh is empty, i.e., has no vertices
        inline bool is_empty() const { return n_vertices() == 0; }

        //! clear mesh: remove all vertices, edges, faces
        virtual void clear();

        //! remove unused memory from vectors
        void free_memory();

        //! reserve memory (mainly used in file readers)
        void reserve(size_t nvertices, size_t nedges, size_t nfaces);

        //! remove deleted elements
        void garbage_collection();

        //! \return whether vertex \p v is deleted
        //! \sa garbage_collection()
        inline bool is_deleted(Vertex v) const { return vdeleted_[v]; }

        //! \return whether halfedge \p h is deleted
        //! \sa garbage_collection()
        inline bool is_deleted(Halfedge h) const { return edeleted_[edge(h)]; }

        //! \return whether edge \p e is deleted
        //! \sa garbage_collection()
        inline bool is_deleted(Edge e) const { return edeleted_[e]; }

        //! \return whether face \p f is deleted
        //! \sa garbage_collection()
        inline bool is_deleted(Face f) const { return fdeleted_[f]; }

        //! \return whether vertex \p v is valid.
        inline bool is_valid(Vertex v) const { return (v.idx() < vertices_size()); }

        //! \return whether halfedge \p h is valid.
        inline bool is_valid(Halfedge h) const { return (h.idx() < halfedges_size()); }

        //! \return whether edge \p e is valid.
        inline bool is_valid(Edge e) const { return (e.idx() < edges_size()); }

        //! \return whether the face \p f is valid.
        inline bool is_valid(Face f) const { return (f.idx() < faces_size()); }

        //!@}
        //! \name Low-level connectivity
        //!@{

        //! \return an outgoing halfedge of vertex \p v.
        //! if \p v is a boundary vertex this will be a boundary halfedge.
        inline Halfedge halfedge(Vertex v) const { return vconn_[v].halfedge_; }

        //! set the outgoing halfedge of vertex \p v to \p h
        inline void set_halfedge(Vertex v, Halfedge h) { vconn_[v].halfedge_ = h; }

        //! \return whether \p v is a boundary vertex
        inline bool is_boundary(Vertex v) const {
            Halfedge h(halfedge(v));
            return (!(h.is_valid() && face(h).is_valid()));
        }

        //! \return whether \p v is isolated, i.e., not incident to any edge
        inline bool is_isolated(Vertex v) const { return !halfedge(v).is_valid(); }

        //! \return whether \p v is a manifold vertex (not incident to several patches)
        bool is_manifold(Vertex v) const {
            // The vertex is non-manifold if more than one gap exists, i.e.
            // more than one outgoing boundary halfedge.
            int n(0);
            auto hit = halfedges(v);
            auto hend = hit;
            if (hit)
                do {
                    if (is_boundary(*hit))
                        ++n;
                } while (++hit != hend);
            return n < 2;
        }

        //! \return the vertex the halfedge \p h points to
        inline Vertex to_vertex(Halfedge h) const { return hconn_[h].vertex_; }

        //! \return the vertex the halfedge \p h emanates from
        inline Vertex from_vertex(Halfedge h) const {
            return to_vertex(opposite_halfedge(h));
        }

        //! sets the vertex the halfedge \p h points to to \p v
        inline void set_vertex(Halfedge h, Vertex v) { hconn_[h].vertex_ = v; }

        //! \return the face incident to halfedge \p h
        inline Face face(Halfedge h) const { return hconn_[h].face_; }

        //! sets the incident face to halfedge \p h to \p f
        inline void set_face(Halfedge h, Face f) { hconn_[h].face_ = f; }

        //! \return the next halfedge within the incident face
        inline Halfedge next_halfedge(Halfedge h) const {
            return hconn_[h].next_halfedge_;
        }

        //! sets the next halfedge of \p h within the face to \p nh
        inline void set_next_halfedge(Halfedge h, Halfedge nh) {
            hconn_[h].next_halfedge_ = nh;
            hconn_[nh].prev_halfedge_ = h;
        }

        //! sets the previous halfedge of \p h and the next halfedge of \p ph to \p nh
        inline void set_prev_halfedge(Halfedge h, Halfedge ph) {
            hconn_[h].prev_halfedge_ = ph;
            hconn_[ph].next_halfedge_ = h;
        }

        //! \return the previous halfedge within the incident face
        inline Halfedge prev_halfedge(Halfedge h) const {
            return hconn_[h].prev_halfedge_;
        }

        //! \return the opposite halfedge of \p h
        inline Halfedge opposite_halfedge(Halfedge h) const {
            return Halfedge((h.idx() & 1) ? h.idx() - 1 : h.idx() + 1);
        }

        //! \return the halfedge that is rotated counter-clockwise around the
        //! start vertex of \p h. it is the opposite halfedge of the previous
        //! halfedge of \p h.
        inline Halfedge ccw_rotated_halfedge(Halfedge h) const {
            return opposite_halfedge(prev_halfedge(h));
        }

        //! \return the halfedge that is rotated clockwise around the start
        //! vertex of \p h. it is the next halfedge of the opposite halfedge of
        //! \p h.
        inline Halfedge cw_rotated_halfedge(Halfedge h) const {
            return next_halfedge(opposite_halfedge(h));
        }

        //! \return the edge that contains halfedge \p h as one of its two
        //! halfedges.
        inline Edge edge(Halfedge h) const { return Edge(h.idx() >> 1); }

        //! \return whether h is a boundary halfedge, i.e., if its face does not exist.
        inline bool is_boundary(Halfedge h) const { return !face(h).is_valid(); }

        //! \return the \p i'th halfedge of edge \p e. \p i has to be 0 or 1.
        inline Halfedge halfedge(Edge e, unsigned int i) const {
            assert(i <= 1);
            return Halfedge((e.idx() << 1) + i);
        }

        //! \return the \p i'th vertex of edge \p e. \p i has to be 0 or 1.
        inline Vertex vertex(Edge e, unsigned int i) const {
            assert(i <= 1);
            return to_vertex(halfedge(e, i));
        }

        //! \return the face incident to the \p i'th halfedge of edge \p e. \p i has to be 0 or 1.
        inline Face face(Edge e, unsigned int i) const {
            assert(i <= 1);
            return face(halfedge(e, i));
        }

        //! \return whether \p e is a boundary edge, i.e., if one of its
        //! halfedges is a boundary halfedge.
        inline bool is_boundary(Edge e) const {
            return (is_boundary(halfedge(e, 0)) || is_boundary(halfedge(e, 1)));
        }

        //! \return a halfedge of face \p f
        inline Halfedge halfedge(Face f) const { return fconn_[f].halfedge_; }

        //! sets the halfedge of face \p f to \p h
        inline void set_halfedge(Face f, Halfedge h) { fconn_[f].halfedge_ = h; }

        //! \return whether \p f is a boundary face, i.e., it one of its edges is a boundary edge.
        bool is_boundary(Face f) const {
            Halfedge h = halfedge(f);
            Halfedge hh = h;
            do {
                if (is_boundary(opposite_halfedge(h)))
                    return true;
                h = next_halfedge(h);
            } while (h != hh);
            return false;
        }

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

        //! \return the names of all halfedge properties
        inline std::vector<std::string> halfedge_properties() const {
            return hprops_.properties();
        }

        //! \return the names of all edge properties
        inline std::vector<std::string> edge_properties() const {
            return eprops_.properties();
        }

        //! add a face property of type \p T with name \p name and default value \c
        //! t.  fails if a property named \p name exists already, since the name has
        //! to be unique.  in this case it returns an invalid property
        template<class T>
        FaceProperty<T> add_face_property(const std::string &name, const T t = T()) {
            return FaceProperty<T>(fprops_.add<T>(name, t));
        }

        //! get the face property named \p name of type \p T. returns an invalid
        //! VertexProperty if the property does not exist or if the type does not
        //! match.
        template<class T>
        FaceProperty<T> get_face_property(const std::string &name) const {
            return FaceProperty<T>(fprops_.get<T>(name));
        }

        //! if a face property of type \p T with name \p name exists, it is
        //! returned.  otherwise this property is added (with default value \p t)
        template<class T>
        FaceProperty<T> face_property(const std::string &name, const T t = T()) {
            return FaceProperty<T>(fprops_.get_or_add<T>(name, t));
        }

        //! remove the face property \p p
        template<class T>
        void remove_face_property(FaceProperty<T> &p) {
            fprops_.remove(p);
        }

        //! does the mesh have a face property with name \p name?
        bool has_face_property(const std::string &name) const {
            return fprops_.exists(name);
        }

        //! \return the names of all face properties
        std::vector<std::string> face_properties() const {
            return fprops_.properties();
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

        //! \return vertex container for C++11 range-based for-loops
        VertexContainer vertices() const {
            return VertexContainer(vertices_begin(), vertices_end());
        }

        //! \return start iterator for halfedges
        HalfedgeIterator halfedges_begin() const {
            return HalfedgeIterator(Halfedge(0), this);
        }

        //! \return end iterator for halfedges
        HalfedgeIterator halfedges_end() const {
            return HalfedgeIterator(
                    Halfedge(static_cast<IndexType>(halfedges_size())), this);
        }

        //! \return halfedge container for C++11 range-based for-loops
        HalfedgeContainer halfedges() const {
            return HalfedgeContainer(halfedges_begin(), halfedges_end());
        }

        //! \return start iterator for edges
        EdgeIterator edges_begin() const { return EdgeIterator(Edge(0), this); }

        //! \return end iterator for edges
        EdgeIterator edges_end() const {
            return EdgeIterator(Edge(static_cast<IndexType>(edges_size())), this);
        }

        //! \return edge container for C++11 range-based for-loops
        EdgeContainer edges() const {
            return EdgeContainer(edges_begin(), edges_end());
        }

        //! \return circulator for vertices around vertex \p v
        VertexAroundVertexCirculator vertices(Vertex v) const {
            return VertexAroundVertexCirculator(this, v);
        }

        //! \return circulator for edges around vertex \p v
        EdgeAroundVertexCirculator edges(Vertex v) const {
            return EdgeAroundVertexCirculator(this, v);
        }

        //! \return circulator for outgoing halfedges around vertex \p v
        HalfedgeAroundVertexCirculator halfedges(Vertex v) const {
            return HalfedgeAroundVertexCirculator(this, v);
        }

        //! \return start iterator for faces
        FaceIterator faces_begin() const { return FaceIterator(Face(0), this); }

        //! \return end iterator for faces
        FaceIterator faces_end() const {
            return FaceIterator(Face(static_cast<IndexType>(faces_size())), this);
        }

        //! \return face container for C++11 range-based for-loops
        FaceContainer faces() const {
            return FaceContainer(faces_begin(), faces_end());
        }

        //! \return circulator for faces around vertex \p v
        FaceAroundVertexCirculator faces(Vertex v) const {
            return FaceAroundVertexCirculator(this, v);
        }

        //! \return circulator for vertices of face \p f
        VertexAroundFaceCirculator vertices(Face f) const {
            return VertexAroundFaceCirculator(this, f);
        }

        //! \return circulator for halfedges of face \p f
        HalfedgeAroundFaceCirculator halfedges(Face f) const {
            return HalfedgeAroundFaceCirculator(this, f);
        }

        //!@}
        //! \name Higher-level Topological Operations
        //!@{

        //! Subdivide the edge \p e = (v0,v1) by splitting it into the two edge
        //! (v0,p) and (p,v1). Note that this function does not introduce any
        //! other edge or faces. It simply splits the edge. Returns halfedge that
        //! points to \p p.
        //! \sa insert_vertex(Edge, Vertex)
        //! \sa insert_vertex(Halfedge, Vertex)
        inline Halfedge insert_vertex(Edge e, const PointType &p) {
            return insert_vertex(halfedge(e, 0), add_vertex(p));
        }

        //! Subdivide the edge \p e = (v0,v1) by splitting it into the two edge
        //! (v0,v) and (v,v1). Note that this function does not introduce any
        //! other edge or faces. It simply splits the edge. Returns halfedge
        //! that points to \p p. \sa insert_vertex(Edge, Point) \sa
        //! insert_vertex(Halfedge, Vertex)
        inline Halfedge insert_vertex(Edge e, Vertex v) {
            return insert_vertex(halfedge(e, 0), v);
        }

        //! Subdivide the halfedge \p h = (v0,v1) by splitting it into the two halfedges
        //! (v0,v) and (v,v1). Note that this function does not introduce any
        //! other edge or faces. It simply splits the edge. Returns the halfedge
        //! that points from v1 to \p v.
        //! \sa insert_vertex(Edge, Point)
        //! \sa insert_vertex(Edge, Vertex)
        Halfedge insert_vertex(Halfedge h0, Vertex v);

        //! find the halfedge from start to end
        Halfedge find_halfedge(Vertex start, Vertex end) const;

        //! find the edge (a,b)
        Edge find_edge(Vertex a, Vertex b) const;

        //! \return whether the mesh a triangle mesh. this function simply tests
        //! each face, and therefore is not very efficient.
        bool is_triangle_mesh() const;

        //! \return whether the mesh a quad mesh. this function simply tests
        //! each face, and therefore is not very efficient.
        bool is_quad_mesh() const;

        //! \return whether collapsing the halfedge \p v0v1 is topologically legal.
        //! \attention This function is only valid for triangle meshes.
        bool is_collapse_ok(Halfedge v0v1) const;

        //! Collapse the halfedge \p h by moving its start vertex into its target
        //! vertex. For non-boundary halfedges this function removes one vertex, three
        //! edges, and two faces. For boundary halfedges it removes one vertex, two
        //! edges and one face.
        //! \attention This function is only valid for triangle meshes.
        //! \attention Halfedge collapses might lead to invalid faces. Call
        //! is_collapse_ok(Halfedge) to be sure the collapse is legal.
        //! \attention The removed items are only marked as deleted. You have
        //! to call garbage_collection() to finally remove them.
        void collapse(Halfedge h);

        //! \return whether removing the edge \p e is topologically legal.
        bool is_removal_ok(Edge e) const;

        //! Remove edge and merge its two incident faces into one.
        //! This operation requires that the edge has two incident faces
        //! and that these two are not equal.
        //! \sa is_removal_ok(Edge)
        bool remove_edge(Edge e);

        //! Split the face \p f by first adding point \p p to the mesh and then
        //! inserting edges between \p p and the vertices of \p f. For a triangle
        //! this is a standard one-to-three split.
        //! \sa split(Face, Vertex)
        inline Vertex split(Face f, const PointType &p) {
            Vertex v = add_vertex(p);
            split(f, v);
            return v;
        }

        //! Split the face \p f by inserting edges between \p v and the vertices
        //! of \p f. For a triangle this is a standard one-to-three split.
        //! \sa split(Face, const Point&)
        void split(Face f, Vertex v);

        //! Split the edge \p e by first adding point \p p to the mesh and then
        //! connecting it to the two vertices of the adjacent triangles that are
        //! opposite to edge \p e. Returns the halfedge pointing to \p p that is
        //! created by splitting the existing edge \p e.
        //!
        //! \attention This function is only valid for triangle meshes.
        //! \sa split(Edge, Vertex)
        inline Halfedge split(Edge e, const PointType &p) { return split(e, add_vertex(p)); }

        //! Split the edge \p e by connecting vertex \p v it to the two
        //! vertices of the adjacent triangles that are opposite to edge \c
        //! e. Returns the halfedge pointing to \p v that is created by splitting
        //! the existing edge \p e.
        //!
        //! \attention This function is only valid for triangle meshes.
        //! \sa split(Edge, const Point&)
        Halfedge split(Edge e, Vertex v);

        //! Insert edge between the to-vertices of \p h0 and \p h1.
        //! \return The new halfedge from v0 to v1.
        //! \attention \p h0 and \p h1 have to belong to the same face.
        Halfedge insert_edge(Halfedge h0, Halfedge h1);

        //! Check whether flipping edge \p e is topologically OK.
        //! \attention This function is only valid for triangle meshes.
        //! \sa flip(Edge)
        bool is_flip_ok(Edge e) const;

        //! Flip the edge \p e . Removes the edge \p e and add an edge between the
        //! two vertices opposite to edge \p e of the two incident triangles.
        //! \attention This function is only valid for triangle meshes.
        //! \attention Flipping an edge may result in a non-manifold mesh, hence check
        //! for yourself whether this operation is allowed or not!
        //! \sa is_flip_ok()
        void flip(Edge e);

        //! Compute the valence of vertex \p v (number of incident edges).
        size_t valence(Vertex v) const;

        //! Compute the valence of face \p f (its number of vertices).
        size_t valence(Face f) const;

        //! Delete vertex \p v from the mesh.
        //! \note Only marks the vertex as deleted. Call garbage_collection() to finally remove deleted entities.
        void delete_vertex(Vertex v);

        //! Delete edge \p e from the mesh.
        //! \note Only marks the edge as deleted. Call garbage_collection() to finally remove deleted entities.
        void delete_edge(Edge e);

        //! Deletes face \p f from the mesh.
        //! \note Only marks the face as deleted. Call garbage_collection() to finally remove deleted entities.
        void delete_face(Face f);

        //!@}
        //! \name Geometry-related Functions
        //!@{

        //! position of a vertex (read only)
        inline const PointType &position(Vertex v) const { return vpoint_[v]; }

        //! position of a vertex
        inline PointType &position(Vertex v) { return vpoint_[v]; }

        //! \return vector of point positions
        inline std::vector<PointType> &positions() { return vpoint_.vector(); }

        //!@}

        //! \name Allocate new elements
        //!@{

        //! \brief Allocate a new vertex, resize vertex properties accordingly.
        //! \throw AllocationException in case of failure to allocate a new vertex.
        inline Vertex new_vertex() {
            if (vertices_size() == BCG_MAX_INDEX - 1) {
                auto what =
                        "SurfaceMesh: cannot allocate vertex, max. index reached";
                throw AllocationException(what);
            }
            vprops_.push_back();
            return Vertex(static_cast<IndexType>(vertices_size()) - 1);
        }

        //! \brief Allocate a new edge, resize edge and halfedge properties accordingly.
        //! \throw AllocationException in case of failure to allocate a new edge.
        inline Halfedge new_edge() {
            if (halfedges_size() == BCG_MAX_INDEX - 1) {
                auto what = "SurfaceMesh: cannot allocate edge, max. index reached";
                throw AllocationException(what);
            }

            eprops_.push_back();
            hprops_.push_back();
            hprops_.push_back();

            Halfedge h0(static_cast<IndexType>(halfedges_size()) - 2);
            Halfedge h1(static_cast<IndexType>(halfedges_size()) - 1);

            return h0;
        }

        //! \brief Allocate a new edge, resize edge and halfedge properties accordingly.
        //! \throw AllocationException in case of failure to allocate a new edge.
        //! \param start starting Vertex of the new edge
        //! \param end end Vertex of the new edge
        inline Halfedge new_edge(Vertex start, Vertex end) {
            assert(start != end);

            if (halfedges_size() == BCG_MAX_INDEX - 1) {
                auto what = "SurfaceMesh: cannot allocate edge, max. index reached";
                throw AllocationException(what);
            }

            eprops_.push_back();
            hprops_.push_back();
            hprops_.push_back();

            Halfedge h0(static_cast<IndexType>(halfedges_size()) - 2);
            Halfedge h1(static_cast<IndexType>(halfedges_size()) - 1);

            set_vertex(h0, end);
            set_vertex(h1, start);

            return h0;
        }

        //! \brief Allocate a new face, resize face properties accordingly.
        //! \throw AllocationException in case of failure to allocate a new face.
        inline Face new_face() {
            if (faces_size() == BCG_MAX_INDEX - 1) {
                auto what = "SurfaceMesh: cannot allocate face, max. index reached";
                throw AllocationException(what);
            }

            fprops_.push_back();
            return Face(static_cast<IndexType>(faces_size()) - 1);
        }

        //!@}

        struct VertexConnectivity {
            // an outgoing halfedge per vertex (it will be a boundary halfedge
            // for boundary vertices)
            Halfedge halfedge_;

            friend std::ostream &operator<<(std::ostream &os, const VertexConnectivity &vc) {
                os << "h: " << vc.halfedge_.idx_;
                return os;
            }
        };

        struct HalfedgeConnectivity {
            Face face_;              // face incident to halfedge
            Vertex vertex_;          // vertex the halfedge points to
            Halfedge next_halfedge_; // next halfedge
            Halfedge prev_halfedge_; // previous halfedge

            friend std::ostream &operator<<(std::ostream &os, const HalfedgeConnectivity &hc) {
                os << "f: " << hc.face_.idx_
                   << "v: " << hc.vertex_.idx_
                   << "nh: " << hc.next_halfedge_.idx_
                   << "ph: " << hc.prev_halfedge_.idx_;
                return os;
            }
        };

        struct FaceConnectivity {
            Halfedge halfedge_; // a halfedge that is part of the face

            friend std::ostream &operator<<(std::ostream &os, const FaceConnectivity &fc) {
                os << "h: " << fc.halfedge_.idx_;
                return os;
            }
        };

        // make sure that the outgoing halfedge of vertex \p v is a boundary
        // halfedge if \p v is a boundary vertex.
        void adjust_outgoing_halfedge(Vertex v);

        // Helper for halfedge collapse
        void remove_edge_helper(Halfedge h);

        // Helper for halfedge collapse
        void remove_loop_helper(Halfedge h);

        // are there any deleted entities?
        inline bool has_garbage() const { return has_garbage_; }

        // io functions that need access to internal details
        friend void read_pmp(SurfaceMesh &, const std::filesystem::path &);

        friend void write_pmp(const SurfaceMesh &, const std::filesystem::path &,
                              const IOFlags &);

        // property containers for each entity type and object
        PropertyContainer vprops_;
        PropertyContainer hprops_;
        PropertyContainer eprops_;
        PropertyContainer fprops_;

        // point coordinates
        VertexProperty<PointType> vpoint_;

        // connectivity information
        VertexProperty<VertexConnectivity> vconn_;
        HalfedgeProperty<HalfedgeConnectivity> hconn_;
        FaceProperty<FaceConnectivity> fconn_;

        // markers for deleted entities
        VertexProperty<bool> vdeleted_;
        EdgeProperty<bool> edeleted_;
        FaceProperty<bool> fdeleted_;

        // numbers of deleted entities
        IndexType deleted_vertices_{0};
        IndexType deleted_edges_{0};
        IndexType deleted_faces_{0};

        // indicate garbage present
        bool has_garbage_{false};

        // helper data for add_face()
        using NextCacheEntry = std::pair<Halfedge, Halfedge>;
        using NextCache = std::vector<NextCacheEntry>;
        std::vector<Vertex> add_face_vertices_;
        std::vector<Halfedge> add_face_halfedges_;
        std::vector<bool> add_face_is_new_;
        std::vector<bool> add_face_needs_adjust_;
        NextCache add_face_next_cache_;
    };

//!@}

} // namespace pmp
