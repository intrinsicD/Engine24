//
// Created by alex on 30.07.24.
//

#include "PointCloud.h"

namespace Bcg {
    PointCloud::PointCloud() {
        // allocate standard properties
        // same list is used in operator=() and assign()
        vpoint_ = add_vertex_property<PointType>("v:point");

        vdeleted_ = add_vertex_property<bool>("v:deleted", false);
    }

    //! destructor
    PointCloud::~PointCloud() = default;

    //! assign \p rhs to \p *this. performs a deep copy of all properties.
    PointCloud &PointCloud::operator=(const PointCloud &rhs) {
        if (this != &rhs) {
            // deep copy of property containers
            vprops_ = rhs.vprops_;

            // property handles contain pointers, have to be reassigned
            vpoint_ = vertex_property<PointType>("v:point");

            vdeleted_ = vertex_property<bool>("v:deleted");

            // how many elements are deleted?
            deleted_vertices_ = rhs.deleted_vertices_;

            has_garbage_ = rhs.has_garbage_;
        }

        return *this;
    }

    //! assign \p rhs to \p *this. does not copy custom properties.
    PointCloud &PointCloud::assign(const PointCloud &rhs) {
        if (this != &rhs) {
            // clear properties
            vprops_.clear();

            // allocate standard properties
            vpoint_ = add_vertex_property<PointType>("v:point");

            vdeleted_ = add_vertex_property<bool>("v:deleted", false);

            // copy properties from other mesh
            vpoint_.array() = rhs.vpoint_.array();

            vdeleted_.array() = rhs.vdeleted_.array();

            // resize (needed by property containers)
            vprops_.resize(rhs.vertices_size());

            // how many elements are deleted?
            deleted_vertices_ = rhs.deleted_vertices_;
            has_garbage_ = rhs.has_garbage_;
        }

        return *this;
    }

    //! add a new vertex with position \p p
    Vertex PointCloud::add_vertex(const PointType &p) {
        Vertex v = new_vertex();
        if (v.is_valid())
            vpoint_[v] = p;
        return v;
    }

    //! clear mesh: remove all vertices, edges, faces
    void PointCloud::clear() {
// remove all properties
        vprops_.clear();

        // really free their memory
        free_memory();

        // add the standard properties back
        vpoint_ = add_vertex_property<PointType>("v:point");

        vdeleted_ = add_vertex_property<bool>("v:deleted", false);

        // set initial status (as in constructor)
        deleted_vertices_ = 0;

        has_garbage_ = false;
    }

    //! remove unused memory from vectors
    void PointCloud::free_memory() {
        vprops_.free_memory();
    }

    //! reserve memory (mainly used in file readers)
    void PointCloud::reserve(size_t nvertices, size_t nedges, size_t nfaces) {
        vprops_.reserve(nvertices);
    }

    void PointCloud::garbage_collection() {
        if (!has_garbage_)
            return;

        auto nV = vertices_size();

        // setup handle mapping
        VertexProperty<Vertex> vmap =
                add_vertex_property<Vertex>("v:garbage-collection");

        for (size_t i = 0; i < nV; ++i)
            vmap[Vertex(i)] = Vertex(i);

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

        // remove handle maps
        remove_vertex_property(vmap);

        // finally resize arrays
        vprops_.resize(nV);
        vprops_.free_memory();

        deleted_vertices_ = 0;
        has_garbage_ = false;
    }
}