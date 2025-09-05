//
// Created by alex on 26.08.24.
//

#include "Graph.h"

namespace Bcg{
    Graph::Graph(){
        vpoint_ = add_vertex_property<PointType>("v:point");
        vconn_ = add_vertex_property<Halfedge>("v:connectivity");
        hconn_ = add_halfedge_property<HalfedgeConnectivity>("h:connectivity");

        vdeleted_ = add_vertex_property<bool>("v:deleted", false);
        edeleted_ = add_edge_property<bool>("e:deleted", false);
    }

    Graph &Graph::operator=(const Graph &rhs){
        if (this != &rhs) {
            // deep copy of property containers
            vprops_ = rhs.vprops_;
            hprops_ = rhs.hprops_;
            eprops_ = rhs.eprops_;

            // property handles contain pointers, have to be reassigned
            vpoint_ = vertex_property<PointType>("v:point");
            vconn_ = vertex_property<Halfedge>("v:connectivity");
            hconn_ = halfedge_property<HalfedgeConnectivity>("h:connectivity");

            vdeleted_ = vertex_property<bool>("v:deleted");
            edeleted_ = edge_property<bool>("e:deleted");

            // how many elements are deleted?
            deleted_vertices_ = rhs.deleted_vertices_;
            deleted_edges_ = rhs.deleted_edges_;

            has_garbage_ = rhs.has_garbage_;
        }

        return *this;
    }

    Graph &Graph::assign(const Graph &rhs){
        if (this != &rhs) {
            // clear properties
            vprops_.clear();
            hprops_.clear();
            eprops_.clear();

            // allocate standard properties
            vpoint_ = add_vertex_property<PointType>("v:point");
            vconn_ = add_vertex_property<Halfedge>("v:connectivity");
            hconn_ = add_halfedge_property<HalfedgeConnectivity>("h:connectivity");

            vdeleted_ = add_vertex_property<bool>("v:deleted", false);
            edeleted_ = add_edge_property<bool>("e:deleted", false);

            // copy properties from other mesh
            vpoint_.array() = rhs.vpoint_.array();
            vconn_.array() = rhs.vconn_.array();
            hconn_.array() = rhs.hconn_.array();

            vdeleted_.array() = rhs.vdeleted_.array();
            edeleted_.array() = rhs.edeleted_.array();

            // resize (needed by property containers)
            vprops_.resize(rhs.vertices_size());
            hprops_.resize(rhs.halfedges_size());
            eprops_.resize(rhs.edges_size());

            // how many elements are deleted?
            deleted_vertices_ = rhs.deleted_vertices_;
            deleted_edges_ = rhs.deleted_edges_;
            has_garbage_ = rhs.has_garbage_;
        }

        return *this;
    }

    Vertex Graph::add_vertex(const PointType &p){
        Vertex v = new_vertex();
        if (v.is_valid())
            vpoint_[v] = p;
        return v;
    }

    void Graph::mark_vertex_deleted(Vertex v) {
        if (!vprops_.is_valid(v)) return;
        if (!vdeleted_[v]){
            vdeleted_[v] = true;
            ++deleted_vertices_;
        }

        has_garbage_ = true;
    }

    void Graph::remove_vertex(Vertex v) {
        if (!vprops_.is_valid(v)) return;
        if (vdeleted_[v]) return;

        for (const auto &h: get_halfedges(v)) {
            remove_edge(get_edge(h));
        }

        mark_vertex_deleted(v);
    }

    void Graph::clear(){
        // remove all properties
        vprops_.clear();
        hprops_.clear();
        eprops_.clear();

        // really free their memory
        free_memory();

        // add the standard properties back
        vpoint_ = add_vertex_property<PointType>("v:point");
        vconn_ = add_vertex_property<Halfedge>("v:connectivity");
        hconn_ = add_halfedge_property<HalfedgeConnectivity>("h:connectivity");
        vdeleted_ = add_vertex_property<bool>("v:deleted", false);
        edeleted_ = add_edge_property<bool>("e:deleted", false);

        // set initial status (as in constructor)
        deleted_vertices_ = 0;
        deleted_edges_ = 0;
        has_garbage_ = false;
    }

    void Graph::free_memory(){
        vprops_.free_memory();
        hprops_.free_memory();
        eprops_.free_memory();
    }

    void Graph::reserve(size_t nvertices, size_t nedges){
        vprops_.reserve(nvertices);
        hprops_.reserve(2 * nedges);
        eprops_.reserve(nedges);
    }

    void Graph::garbage_collection(){
        if (!has_garbage_)
            return;

        auto nV = vertices_size();
        auto nE = edges_size();
        auto nH = halfedges_size();

        // setup handle mapping
        VertexProperty<Vertex> vmap =
                add_vertex_property<Vertex>("v:garbage-collection");
        HalfedgeProperty<Halfedge> hmap =
                add_halfedge_property<Halfedge>("h:garbage-collection");

        for (size_t i = 0; i < nV; ++i)
            vmap[Vertex(i)] = Vertex(i);
        for (size_t i = 0; i < nH; ++i)
            hmap[Halfedge(i)] = Halfedge(i);

        // remove deleted vertices
        if (nV > 0) {
            size_t i0 = 0;
            size_t i1 = nV - 1;

            while (true) {
                // find first deleted and last un-deleted
                while (!vdeleted_[Vertex(i0)] && i0 < i1)
                    ++i0;
                while (vdeleted_[Vertex(i1)] && i0 < i1)
                    --i1;
                if (i0 >= i1)
                    break;

                // swap
                vprops_.swap(i0, i1);
            }

            // remember new size
            nV = vdeleted_[Vertex(i0)] ? i0 : i0 + 1;
        }

        // remove deleted edges
        if (nE > 0) {
            size_t i0 = 0;
            size_t i1 = nE - 1;

            while (true) {
                // find first deleted and last un-deleted
                while (!edeleted_[Edge(i0)] && i0 < i1)
                    ++i0;
                while (edeleted_[Edge(i1)] && i0 < i1)
                    --i1;
                if (i0 >= i1)
                    break;

                // swap
                eprops_.swap(i0, i1);
                hprops_.swap(2 * i0, 2 * i1);
                hprops_.swap(2 * i0 + 1, 2 * i1 + 1);
            }

            // remember new size
            nE = edeleted_[Edge(i0)] ? i0 : i0 + 1;
            nH = 2 * nE;
        }

        // update vertex connectivity
        for (size_t i = 0; i < nV; ++i) {
            auto v = Vertex(i);
            if (!is_isolated(v))
                set_halfedge(v, hmap[get_halfedge(v)]);
        }

        // update halfedge connectivity
        for (size_t i = 0; i < nH; ++i) {
            auto h = Halfedge(i);
            set_vertex(h, vmap[to_vertex(h)]);
            set_next(h, hmap[get_next(h)]);
        }

        // remove handle maps
        remove_vertex_property(vmap);
        remove_halfedge_property(hmap);

        // finally resize arrays
        vprops_.resize(nV);
        vprops_.free_memory();
        hprops_.resize(nH);
        hprops_.free_memory();
        eprops_.resize(nE);
        eprops_.free_memory();

        deleted_vertices_ = deleted_edges_ = 0;
        has_garbage_ = false;
    }

    Property<Vector<IndexType, 2>> Graph::get_edges(){
        auto indices = edge_property<Vector<IndexType, 2>>("e:indices");
        for (auto e: edges()) {
            indices[e] = {to_vertex(get_halfedge(e, 0)).idx(), to_vertex(get_halfedge(e, 1)).idx()};
        }
        return indices;
    }

    Halfedge Graph::find_halfedge(Vertex v0, Vertex v1) const{
        if (!is_valid(get_halfedge(v0))) {
            return {};
        }
        for (auto h: get_halfedges(v0)) {
            if (to_vertex(h) == v1) {
                return h;
            }
        }
        return {};
    }


    Halfedge Graph::new_edge(Vertex v0, Vertex v1) {
        if (v0 == v1 || !v0.is_valid() || !v1.is_valid()) {
            assert(v0 != v1);
            return {};
        }

        eprops_.push_back();
        hprops_.push_back();
        hprops_.push_back();

        Halfedge h(hprops_.size() - 2);
        Halfedge o(hprops_.size() - 1);

        set_vertex(h, v1);
        set_vertex(o, v0);

        set_next(h, o);
        set_next(o, h);

        return h;
    }

    Halfedge Graph::add_edge(Vertex v0, Vertex v1) {
        if (v1 == v0 || !v0.is_valid() || !v1.is_valid()) {
            assert(v0 != v1);
            return {};
        }
        Halfedge h01 = find_halfedge(v0, v1);
        if (is_valid(h01)) {
            return h01;
        }

        Halfedge h = new_edge(v0, v1);
        Halfedge o = get_opposite(h);

        Halfedge in_0 = get_opposite(get_halfedge(v0));
        if (is_valid(in_0)) {
            Halfedge out_next_0 = get_next(in_0);
            set_next(in_0, h);
            set_next(o, out_next_0);
        }
        set_halfedge(v0, h);

        Halfedge out_1 = get_halfedge(v1);
        if (is_valid(out_1)) {
            Halfedge out_prev_1 = get_prev(out_1);
            set_next(h, out_1);
            set_next(out_prev_1, o);
        }
        set_halfedge(v1, o);

        return h;
    }


    void Graph::mark_edge_deleted(Edge e) {
        if (!edeleted_[e]) {
            edeleted_[e] = true;
            ++deleted_edges_;
            mark_halfedge_deleted(get_halfedge(e, 0));
            mark_halfedge_deleted(get_halfedge(e, 1));
        }
        has_garbage_ = true;
    }

    void Graph::remove_edge(Edge e) {
        if (edeleted_[e]) return;

        auto h0 = get_halfedge(e, 0);
        auto h1 = get_halfedge(e, 1);

        auto from_v = to_vertex(h1);
        auto to_v = to_vertex(h0);

        if (hprops_.is_valid(h0)) {
            auto p = get_prev(h0);
            auto n = get_next(h1);
            if (find_halfedge(to_vertex(n), from_v).is_valid()) {
                set_next(p, n);
            }
        }
        if (hprops_.is_valid(h1)) {
            auto p = get_prev(h1);
            auto n = get_next(h0);
            if (find_halfedge(to_vertex(n), to_v).is_valid()) {
                set_next(p, n);
            }
        }

        mark_edge_deleted(e);
    }

    void Graph::mark_halfedge_deleted(Halfedge h) {
        if (!hdeleted_[h]) {
            hdeleted_[h] = true;
            ++deleted_halfedges_;
        }
        has_garbage_ = true;
    }

    size_t Graph::get_valence(Vertex v) const {
        size_t count(0);
        for (const auto &vj: get_vertices(v)) {
            if (vj.is_valid()) {
                ++count;
            }
        }
        return count;
    }
}