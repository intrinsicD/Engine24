//
// Created by alex on 22.08.24.
//

#ifndef ENGINE24_GEOMETRYDATA_H
#define ENGINE24_GEOMETRYDATA_H

#include "Properties.h"
#include "Types.h"
#include "GeometryCommon.h"

namespace Bcg {
    struct Vertices : public PropertyContainer {
        using VertexIterator = Iterator<Vertices, Vertex>;

        Vertices() : deleted_vertices(0) {}

        VertexProperty<bool> vdeleted;
        size_t deleted_vertices = 0;
        bool has_garbage_ = false;

        inline bool is_deleted(Vertex v) const { return vdeleted[v]; }

        inline bool is_valid(Vertex v) const { return (v.idx() < size()); }

        template<class T>
        inline VertexProperty<T> add_vertex_property(const std::string &name,
                                                     const T t = T()) {
            return VertexProperty<T>(add<T>(name, t));
        }

        template<class T>
        inline VertexProperty<T> get_vertex_property(const std::string &name) const {
            return VertexProperty<T>(get<T>(name));
        }

        template<class T>
        inline VertexProperty<T> vertex_property(const std::string &name, const T t = T()) {
            return VertexProperty<T>(get_or_add<T>(name, t));
        }

        template<class T>
        inline void remove_vertex_property(VertexProperty<T> &p) {
            remove(p);
        }

        inline bool has_vertex_property(const std::string &name) const {
            return exists(name);
        }

        inline size_t n_vertices() const { return size() - deleted_vertices; }

        inline bool has_garbage() const { return has_garbage_; }

        inline VertexIterator begin() {
            return {Vertex(0), this};
        }

        inline VertexIterator end() {
            return {Vertex(size()), this};
        }

        inline VertexIterator begin() const {
            return {Vertex(0), this};
        }

        inline VertexIterator end() const {
            return {Vertex(size()), this};
        }
    };

    struct HalfEdges : public PropertyContainer {
        using HalfEdgeIterator = Iterator<HalfEdges, Halfedge>;

        HalfEdges() : deleted_halfedges(0) {}

        HalfedgeProperty<bool> hdeleted;
        size_t deleted_halfedges = 0;

        bool has_garbage_ = false;

        inline bool is_deleted(Halfedge h) const { return hdeleted[h]; }

        inline bool is_valid(Halfedge h) const { return (h.idx() < size()); }

        template<class T>
        inline HalfedgeProperty<T> add_halfedge_property(const std::string &name,
                                                         const T t = T()) {
            return HalfedgeProperty<T>(add<T>(name, t));
        }

        template<class T>
        inline HalfedgeProperty<T> get_halfedge_property(const std::string &name) const {
            return HalfedgeProperty<T>(get<T>(name));
        }


        template<class T>
        inline HalfedgeProperty<T> halfedge_property(const std::string &name, const T t = T()) {
            return HalfedgeProperty<T>(get_or_add<T>(name, t));
        }

        template<class T>
        inline void remove_halfedge_property(HalfedgeProperty<T> &p) {
            remove(p);
        }

        inline bool has_halfedge_property(const std::string &name) const {
            return exists(name);
        }

        inline size_t n_halfedges() const { return size() - deleted_halfedges; }

        inline bool has_garbage() const { return has_garbage_; }

        inline HalfEdgeIterator begin() {
            return {Halfedge(0), this};
        }

        inline HalfEdgeIterator end() {
            return {Halfedge(size()), this};
        }

        inline HalfEdgeIterator begin() const {
            return {Halfedge(0), this};
        }

        inline HalfEdgeIterator end() const {
            return {Halfedge(size()), this};
        }
    };

    struct Edges : public PropertyContainer {
        using EdgeIterator = Iterator<Edges, Edge>;

        Edges() : deleted_edges(0) {}

        EdgeProperty<bool> edeleted;
        size_t deleted_edges = 0;

        bool has_garbage_ = false;

        inline bool is_deleted(Edge e) const { return edeleted[e]; }

        inline bool is_valid(Edge e) const { return (e.idx() < size()); }

        template<class T>
        inline EdgeProperty<T> add_edge_property(const std::string &name,
                                                 const T t = T()) {
            return EdgeProperty<T>(add<T>(name, t));
        }

        template<class T>
        inline EdgeProperty<T> get_edge_property(const std::string &name) const {
            return EdgeProperty<T>(get<T>(name));
        }


        template<class T>
        inline EdgeProperty<T> edge_property(const std::string &name, const T t = T()) {
            return EdgeProperty<T>(get_or_add<T>(name, t));
        }

        template<class T>
        inline void remove_edge_property(EdgeProperty<T> &p) {
            remove(p);
        }

        inline bool has_edge_property(const std::string &name) const {
            return exists(name);
        }

        inline size_t n_edges() const { return size() - deleted_edges; }

        inline bool has_garbage() const { return has_garbage_; }

        inline EdgeIterator begin() {
            return {Edge(0), this};
        }

        inline EdgeIterator end() {
            return {Edge(size()), this};
        }

        inline EdgeIterator begin() const {
            return {Edge(0), this};
        }

        inline EdgeIterator end() const {
            return {Edge(size()), this};
        }
    };

    struct Faces : public PropertyContainer {
        using FaceIterator = Iterator<Faces, Face>;

        Faces() : deleted_faces(0) {}

        FaceProperty<bool> fdeleted;
        size_t deleted_faces = 0;

        bool has_garbage_ = false;

        inline bool is_deleted(Face f) const { return fdeleted[f]; }

        inline bool is_valid(Face f) const { return (f.idx() < size()); }

        template<class T>
        inline FaceProperty<T> add_face_property(const std::string &name,
                                                 const T t = T()) {
            return FaceProperty<T>(add<T>(name, t));
        }

        template<class T>
        inline FaceProperty<T> get_face_property(const std::string &name) const {
            return FaceProperty<T>(get<T>(name));
        }


        template<class T>
        inline FaceProperty<T> face_property(const std::string &name, const T t = T()) {
            return FaceProperty<T>(get_or_add<T>(name, t));
        }

        template<class T>
        inline void remove_face_property(FaceProperty<T> &p) {
            remove(p);
        }

        inline bool has_face_property(const std::string &name) const {
            return exists(name);
        }

        inline size_t n_faces() const { return size() - deleted_faces; }

        inline bool has_garbage() const { return has_garbage_; }

        inline FaceIterator begin() {
            return {Face(0), this};
        }

        inline FaceIterator end() {
            return {Face(size()), this};
        }

        inline FaceIterator begin() const {
            return {Face(0), this};
        }

        inline FaceIterator end() const {
            return {Face(size()), this};
        }
    };

    struct PointCloudData {
        Vertices vertices;
    };

    struct GraphData {
        Vertices vertices;
        HalfEdges halfedges;
        Edges edges;
    };

    struct MeshData {
        Vertices vertices;
        HalfEdges halfedges;
        Edges edges;
        Faces faces;
    };
}

#endif //ENGINE24_GEOMETRYDATA_H
