#pragma once

#include "Properties.h"
#include "GeometryCommon.h"

namespace Bcg {
    struct Vertices : public PropertyContainer {
        using VertexIterator = Iterator<Vertices, Vertex>;

        Vertices()  = default;
        ~Vertices() override = default;

        VertexProperty<bool> vdeleted;
        size_t deleted_vertices = 0;
        bool has_garbage_ = false;

        [[nodiscard]] bool is_deleted(Vertex v) const { return vdeleted[v]; }

        [[nodiscard]] bool is_valid(Vertex v) const { return (v.idx() < size()); }

        template<class T>
        VertexProperty<T> add_vertex_property(const std::string &name,
                                              const T t = T()) {
            return VertexProperty<T>(add<T>(name, t));
        }

        template<class T>
        VertexProperty<T> get_vertex_property(const std::string &name) const {
            return VertexProperty<T>(get<T>(name));
        }

        template<class T>
        VertexProperty<T> vertex_property(const std::string &name, const T t = T()) {
            return VertexProperty<T>(get_or_add<T>(name, t));
        }

        template<class T>
        void remove_vertex_property(VertexProperty<T> &p) {
            remove(p);
        }

        [[nodiscard]] bool has_vertex_property(const std::string &name) const {
            return exists(name);
        }

        [[nodiscard]] size_t n_vertices() const { return size() - deleted_vertices; }

        [[nodiscard]] bool has_garbage() const { return has_garbage_; }

        VertexIterator begin() {
            return {Vertex(0), this};
        }

        VertexIterator end() {
            return {Vertex(size()), this};
        }

        [[nodiscard]] VertexIterator begin() const {
            return {Vertex(0), this};
        }

        [[nodiscard]] VertexIterator end() const {
            return {Vertex(size()), this};
        }
    };

    struct Halfedges : public PropertyContainer {
        using HalfEdgeIterator = Iterator<Halfedges, Halfedge>;

        Halfedges() : deleted_halfedges(0) {
        }

        HalfedgeProperty<bool> hdeleted;
        size_t deleted_halfedges = 0;

        bool has_garbage_ = false;

        [[nodiscard]] bool is_deleted(Halfedge h) const { return hdeleted[h]; }

        [[nodiscard]] bool is_valid(Halfedge h) const { return (h.idx() < size()); }

        template<class T>
        HalfedgeProperty<T> add_halfedge_property(const std::string &name,
                                                  const T t = T()) {
            return HalfedgeProperty<T>(add<T>(name, t));
        }

        template<class T>
        HalfedgeProperty<T> get_halfedge_property(const std::string &name) const {
            return HalfedgeProperty<T>(get<T>(name));
        }


        template<class T>
        HalfedgeProperty<T> halfedge_property(const std::string &name, const T t = T()) {
            return HalfedgeProperty<T>(get_or_add<T>(name, t));
        }

        template<class T>
        void remove_halfedge_property(HalfedgeProperty<T> &p) {
            remove(p);
        }

        [[nodiscard]] bool has_halfedge_property(const std::string &name) const {
            return exists(name);
        }

        [[nodiscard]] size_t n_halfedges() const { return size() - deleted_halfedges; }

        [[nodiscard]] bool has_garbage() const { return has_garbage_; }

        HalfEdgeIterator begin() {
            return {Halfedge(0), this};
        }

        HalfEdgeIterator end() {
            return {Halfedge(size()), this};
        }

        [[nodiscard]] HalfEdgeIterator begin() const {
            return {Halfedge(0), this};
        }

        [[nodiscard]] HalfEdgeIterator end() const {
            return {Halfedge(size()), this};
        }
    };

    struct Edges : public PropertyContainer {
        using EdgeIterator = Iterator<Edges, Edge>;

        Edges() : deleted_edges(0) {
        }

        EdgeProperty<bool> edeleted;
        size_t deleted_edges = 0;

        bool has_garbage_ = false;

        [[nodiscard]] bool is_deleted(Edge e) const { return edeleted[e]; }

        [[nodiscard]] bool is_valid(Edge e) const { return (e.idx() < size()); }

        template<class T>
        EdgeProperty<T> add_edge_property(const std::string &name,
                                          const T t = T()) {
            return EdgeProperty<T>(add<T>(name, t));
        }

        template<class T>
        EdgeProperty<T> get_edge_property(const std::string &name) const {
            return EdgeProperty<T>(get<T>(name));
        }


        template<class T>
        EdgeProperty<T> edge_property(const std::string &name, const T t = T()) {
            return EdgeProperty<T>(get_or_add<T>(name, t));
        }

        template<class T>
        void remove_edge_property(EdgeProperty<T> &p) {
            remove(p);
        }

        [[nodiscard]] bool has_edge_property(const std::string &name) const {
            return exists(name);
        }

        [[nodiscard]] size_t n_edges() const { return size() - deleted_edges; }

        [[nodiscard]] bool has_garbage() const { return has_garbage_; }

        EdgeIterator begin() {
            return {Edge(0), this};
        }

        EdgeIterator end() {
            return {Edge(size()), this};
        }

        [[nodiscard]] EdgeIterator begin() const {
            return {Edge(0), this};
        }

        [[nodiscard]] EdgeIterator end() const {
            return {Edge(size()), this};
        }
    };

    struct Faces : public PropertyContainer {
        using FaceIterator = Iterator<Faces, Face>;

        Faces() : deleted_faces(0) {
        }

        FaceProperty<bool> fdeleted;
        size_t deleted_faces = 0;

        bool has_garbage_ = false;

        [[nodiscard]] bool is_deleted(Face f) const { return fdeleted[f]; }

        [[nodiscard]] bool is_valid(Face f) const { return (f.idx() < size()); }

        template<class T>
        FaceProperty<T> add_face_property(const std::string &name,
                                          const T t = T()) {
            return FaceProperty<T>(add<T>(name, t));
        }

        template<class T>
        FaceProperty<T> get_face_property(const std::string &name) const {
            return FaceProperty<T>(get<T>(name));
        }


        template<class T>
        FaceProperty<T> face_property(const std::string &name, const T t = T()) {
            return FaceProperty<T>(get_or_add<T>(name, t));
        }

        template<class T>
        void remove_face_property(FaceProperty<T> &p) {
            remove(p);
        }

        [[nodiscard]] bool has_face_property(const std::string &name) const {
            return exists(name);
        }

        [[nodiscard]] size_t n_faces() const { return size() - deleted_faces; }

        [[nodiscard]] bool has_garbage() const { return has_garbage_; }

        FaceIterator begin() {
            return {Face(0), this};
        }

        FaceIterator end() {
            return {Face(size()), this};
        }

        [[nodiscard]] FaceIterator begin() const {
            return {Face(0), this};
        }

        [[nodiscard]] FaceIterator end() const {
            return {Face(size()), this};
        }
    };

    struct PointCloudData {
        Vertices vertices;
    };

    struct GraphData {
        Vertices vertices;
        Halfedges halfedges;
        Edges edges;
    };

    struct MeshData {
        Vertices vertices;
        Halfedges halfedges;
        Edges edges;
        Faces faces;
    };
}
