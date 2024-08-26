//
// Created by alex on 26.08.24.
//

#include "Graph.h"

namespace Bcg{
    Graph::Graph(){
        vpoint_ = add_vertex_property<PointType>("v:position");
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
            vpoint_ = vertex_property<PointType>("v:position");
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
            vpoint_ = add_vertex_property<PointType>("v:position");
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

    void Graph::clear(){
        // remove all properties
        vprops_.clear();
        hprops_.clear();
        eprops_.clear();

        // really free their memory
        free_memory();

        // add the standard properties back
        vpoint_ = add_vertex_property<PointType>("v:position");
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
            set_vertex(h, vmap[get_vertex(h)]);
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
            indices[e] = {get_vertex(get_halfedge(e, 0)).idx(), get_vertex(get_halfedge(e, 1)).idx()};
        }
        return indices;
    }

    Halfedge Graph::find_halfedge(Vertex v0, Vertex v1) const{
        if (!is_valid(get_halfedge(v0))) {
            return Halfedge();
        }
        for (auto h: halfedges()) {
            if (get_vertex(h) == v1) {
                return h;
            }
        }
        return {};
    }
}