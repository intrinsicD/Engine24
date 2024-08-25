//
// Created by alex on 25.08.24.
//

#include "MeshInterface.h"

namespace Bcg {

    Vertex HalfedgeMeshInterface::add_vertex(const PointType &p) {
        Vertex v = new_vertex();
        if (v.is_valid())
            vpoint[v] = p;
        return v;
    }

    Face HalfedgeMeshInterface::add_face(const std::vector<Vertex> &vertices) {
        const size_t n(vertices.size());
        assert(n > 2);

        Vertex v;
        size_t i, ii, id;
        Halfedge inner_next, inner_prev, outer_next, outer_prev, boundary_next,
                boundary_prev, patch_start, patch_end;

        // use global arrays to avoid new/delete of local arrays!!!
        std::vector<Halfedge> &halfedges = add_face_halfedges_;
        std::vector<bool> &is_new = add_face_is_new_;
        std::vector<bool> &needs_adjust = add_face_needs_adjust_;
        NextCache &next_cache = add_face_next_cache_;
        halfedges.clear();
        halfedges.resize(n);
        is_new.clear();
        is_new.resize(n);
        needs_adjust.clear();
        needs_adjust.resize(n, false);
        next_cache.clear();
        next_cache.reserve(3 * n);

        // test for topological errors
        for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n) {
            if (!is_boundary(vertices[i])) {
                auto what = "SurfaceMesh::add_face: Complex vertex.";
                throw TopologyException(what);
            }

            halfedges[i] = find_halfedge(vertices[i], vertices[ii]);
            is_new[i] = !halfedges[i].is_valid();

            if (!is_new[i] && !is_boundary(halfedges[i])) {
                auto what = "SurfaceMesh::add_face: Complex edge.";
                throw TopologyException(what);
            }
        }

        // re-link patches if necessary
        for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n) {
            if (!is_new[i] && !is_new[ii]) {
                inner_prev = halfedges[i];
                inner_next = halfedges[ii];

                if (next_halfedge(inner_prev) != inner_next) {
                    // here comes the ugly part... we have to relink a whole patch

                    // search a free gap
                    // free gap will be between boundaryPrev and boundaryNext
                    outer_prev = opposite_halfedge(inner_next);
                    outer_next = opposite_halfedge(inner_prev);
                    boundary_prev = outer_prev;
                    do {
                        boundary_prev =
                                opposite_halfedge(next_halfedge(boundary_prev));
                    } while (!is_boundary(boundary_prev) ||
                             boundary_prev == inner_prev);
                    boundary_next = next_halfedge(boundary_prev);
                    assert(is_boundary(boundary_prev));
                    assert(is_boundary(boundary_next));

                    // ok ?
                    if (boundary_next == inner_next) {
                        auto what =
                                "SurfaceMesh::add_face: Patch re-linking failed.";
                        throw TopologyException(what);
                    }

                    // other halfedges' handles
                    patch_start = next_halfedge(inner_prev);
                    patch_end = prev_halfedge(inner_next);

                    // relink
                    next_cache.emplace_back(boundary_prev, patch_start);
                    next_cache.emplace_back(patch_end, boundary_next);
                    next_cache.emplace_back(inner_prev, inner_next);
                }
            }
        }

        // create missing edges
        for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n) {
            if (is_new[i]) {
                halfedges[i] = new_edge(vertices[i], vertices[ii]);
            }
        }

        // create the face
        Face f(new_face());
        set_halfedge(f, halfedges[n - 1]);

        // setup halfedges
        for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n) {
            v = vertices[ii];
            inner_prev = halfedges[i];
            inner_next = halfedges[ii];

            id = 0;
            if (is_new[i])
                id |= 1;
            if (is_new[ii])
                id |= 2;

            if (id) {
                outer_prev = opposite_halfedge(inner_next);
                outer_next = opposite_halfedge(inner_prev);

                // set outer links
                switch (id) {
                    case 1: // prev is new, next is old
                        boundary_prev = prev_halfedge(inner_next);
                        next_cache.emplace_back(boundary_prev, outer_next);
                        set_halfedge(v, outer_next);
                        break;

                    case 2: // next is new, prev is old
                        boundary_next = next_halfedge(inner_prev);
                        next_cache.emplace_back(outer_prev, boundary_next);
                        set_halfedge(v, boundary_next);
                        break;

                    case 3: // both are new
                        if (!get_halfedge(v).is_valid()) {
                            set_halfedge(v, outer_next);
                            next_cache.emplace_back(outer_prev, outer_next);
                        } else {
                            boundary_next = get_halfedge(v);
                            boundary_prev = prev_halfedge(boundary_next);
                            next_cache.emplace_back(boundary_prev, outer_next);
                            next_cache.emplace_back(outer_prev, boundary_next);
                        }
                        break;
                }

                // set inner link
                next_cache.emplace_back(inner_prev, inner_next);
            } else
                needs_adjust[ii] = (get_halfedge(v) == inner_next);

            // set face handle
            set_face(halfedges[i], f);
        }

        // process next halfedge cache
        NextCache::const_iterator ncIt(next_cache.begin()), ncEnd(next_cache.end());
        for (; ncIt != ncEnd; ++ncIt) {
            set_next_halfedge(ncIt->first, ncIt->second);
        }

        // adjust vertices' halfedge handle
        for (i = 0; i < n; ++i) {
            if (needs_adjust[i]) {
                adjust_outgoing_halfedge(vertices[i]);
            }
        }

        return f;
    }

    Face HalfedgeMeshInterface::add_triangle(Vertex v0, Vertex v1, Vertex v2) {
        add_face_vertices_.resize(3);
        add_face_vertices_[0] = v0;
        add_face_vertices_[1] = v1;
        add_face_vertices_[2] = v2;
        return add_face(add_face_vertices_);
    }

    Face HalfedgeMeshInterface::add_quad(Vertex v0, Vertex v1, Vertex v2, Vertex v3) {
        add_face_vertices_.resize(4);
        add_face_vertices_[0] = v0;
        add_face_vertices_[1] = v1;
        add_face_vertices_[2] = v2;
        add_face_vertices_[3] = v3;
        return add_face(add_face_vertices_);
    }

    void HalfedgeMeshInterface::garbage_collection() {
        if (!vertices.deleted_vertices && !edges.deleted_edges && !faces.deleted_faces)
            return;

        auto nV = vertices.size();
        auto nE = edges.size();
        auto nH = halfedges.size();
        auto nF = faces.size();

        // setup handle mapping
        VertexProperty<Vertex> vmap =
                vertices.add_vertex_property<Vertex>("v:garbage-collection");
        HalfedgeProperty<Halfedge> hmap =
                halfedges.add_halfedge_property<Halfedge>("h:garbage-collection");
        FaceProperty<Face> fmap = faces.add_face_property<Face>("f:garbage-collection");

        for (size_t i = 0; i < nV; ++i)
            vmap[Vertex(i)] = Vertex(i);
        for (size_t i = 0; i < nH; ++i)
            hmap[Halfedge(i)] = Halfedge(i);
        for (size_t i = 0; i < nF; ++i)
            fmap[Face(i)] = Face(i);

        // remove deleted vertices
        if (nV > 0) {
            size_t i0 = 0;
            size_t i1 = nV - 1;

            while (true) {
                // find first deleted and last un-deleted
                while (!vertices.vdeleted[Vertex(i0)] && i0 < i1)
                    ++i0;
                while (vertices.vdeleted[Vertex(i1)] && i0 < i1)
                    --i1;
                if (i0 >= i1)
                    break;

                // swap
                vertices.swap(i0, i1);
            }

            // remember new size
            nV = vertices.vdeleted[Vertex(i0)] ? i0 : i0 + 1;
        }

        // remove deleted edges
        if (nE > 0) {
            size_t i0 = 0;
            size_t i1 = nE - 1;

            while (true) {
                // find first deleted and last un-deleted
                while (!edges.edeleted[Edge(i0)] && i0 < i1)
                    ++i0;
                while (edges.edeleted[Edge(i1)] && i0 < i1)
                    --i1;
                if (i0 >= i1)
                    break;

                // swap
                edges.swap(i0, i1);
                halfedges.swap(2 * i0, 2 * i1);
                halfedges.swap(2 * i0 + 1, 2 * i1 + 1);
            }

            // remember new size
            nE = edges.edeleted[Edge(i0)] ? i0 : i0 + 1;
            nH = 2 * nE;
        }

        // remove deleted faces
        if (nF > 0) {
            size_t i0 = 0;
            size_t i1 = nF - 1;

            while (true) {
                // find 1st deleted and last un-deleted
                while (!faces.fdeleted[Face(i0)] && i0 < i1)
                    ++i0;
                while (faces.fdeleted[Face(i1)] && i0 < i1)
                    --i1;
                if (i0 >= i1)
                    break;

                // swap
                faces.swap(i0, i1);
            }

            // remember new size
            nF = faces.fdeleted[Face(i0)] ? i0 : i0 + 1;
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
            set_next_halfedge(h, hmap[next_halfedge(h)]);
            if (!is_boundary(h))
                set_face(h, fmap[face(h)]);
        }

        // update handles of faces
        for (size_t i = 0; i < nF; ++i) {
            auto f = Face(i);
            set_halfedge(f, hmap[get_halfedge(f)]);
        }

        // remove handle maps
        vertices.remove_vertex_property(vmap);
        halfedges.remove_halfedge_property(hmap);
        faces.remove_face_property(fmap);

        // finally resize arrays
        vertices.resize(nV);
        vertices.free_memory();
        halfedges.resize(nH);
        halfedges.free_memory();
        edges.resize(nE);
        edges.free_memory();
        faces.resize(nF);
        faces.free_memory();

        vertices.deleted_vertices = edges.deleted_edges = faces.deleted_faces = 0;
    }

    bool HalfedgeMeshInterface::is_manifold(Vertex v) const {
        // The vertex is non-manifold if more than one gap exists, i.e.
        // more than one outgoing boundary halfedge.
        int n(0);
        auto hit = get_halfedges(v);
        auto hend = hit;
        if (hit)
            do {
                if (is_boundary(*hit))
                    ++n;
            } while (++hit != hend);
        return n < 2;
    }

    bool HalfedgeMeshInterface::is_boundary(Face f) const {
        Halfedge h = get_halfedge(f);
        Halfedge hh = h;
        do {
            if (is_boundary(opposite_halfedge(h)))
                return true;
            h = next_halfedge(h);
        } while (h != hh);
        return false;
    }

    Halfedge HalfedgeMeshInterface::insert_vertex(Halfedge h0, Vertex v) {
        // before:
        //
        // v0      h0       v2
        //  o--------------->o
        //   <---------------
        //         o0
        //
        // after:
        //
        // v0  h0   v   h1   v2
        //  o------>o------->o
        //   <------ <-------
        //     o0       o1

        Halfedge h2 = next_halfedge(h0);
        Halfedge o0 = opposite_halfedge(h0);
        Halfedge o2 = prev_halfedge(o0);
        Vertex v2 = to_vertex(h0);
        Face fh = face(h0);
        Face fo = face(o0);

        Halfedge h1 = new_edge(v, v2);
        Halfedge o1 = opposite_halfedge(h1);

        // adjust halfedge connectivity
        set_next_halfedge(h1, h2);
        set_next_halfedge(h0, h1);
        set_vertex(h0, v);
        set_vertex(h1, v2);
        set_face(h1, fh);

        set_next_halfedge(o1, o0);
        set_next_halfedge(o2, o1);
        set_vertex(o1, v);
        set_face(o1, fo);

        // adjust vertex connectivity
        set_halfedge(v2, o1);
        adjust_outgoing_halfedge(v2);
        set_halfedge(v, h1);
        adjust_outgoing_halfedge(v);

        // adjust face connectivity
        if (fh.is_valid())
            set_halfedge(fh, h0);
        if (fo.is_valid())
            set_halfedge(fo, o1);

        return o1;
    }

    Halfedge HalfedgeMeshInterface::find_halfedge(Vertex start, Vertex end) const {
        assert(vertices.is_valid(start) && vertices.is_valid(end));

        Halfedge h = get_halfedge(start);
        const Halfedge hh = h;

        if (h.is_valid()) {
            do {
                if (to_vertex(h) == end)
                    return h;
                h = rotate_cw(h);
            } while (h != hh);
        }

        return {};
    }

    Edge HalfedgeMeshInterface::find_edge(Vertex a, Vertex b) const {
        Halfedge h = find_halfedge(a, b);
        return h.is_valid() ? edge(h) : Edge();
    }

    bool HalfedgeMeshInterface::is_triangle_mesh() const {
        for (auto f: faces)
            if (valence(f) != 3)
                return false;

        return true;
    }

    bool HalfedgeMeshInterface::is_quad_mesh() const {
        for (auto f: faces)
            if (valence(f) != 4)
                return false;

        return true;
    }

    bool HalfedgeMeshInterface::is_collapse_ok(Halfedge v0v1) const {
        Halfedge v1v0(opposite_halfedge(v0v1));
        Vertex v0(to_vertex(v1v0));
        Vertex v1(to_vertex(v0v1));
        Vertex vl, vr;
        Halfedge h1, h2;

        // the edges v1-vl and vl-v0 must not be both boundary edges
        if (!is_boundary(v0v1)) {
            vl = to_vertex(next_halfedge(v0v1));
            h1 = next_halfedge(v0v1);
            h2 = next_halfedge(h1);
            if (is_boundary(opposite_halfedge(h1)) &&
                is_boundary(opposite_halfedge(h2)))
                return false;
        }

        // the edges v0-vr and vr-v1 must not be both boundary edges
        if (!is_boundary(v1v0)) {
            vr = to_vertex(next_halfedge(v1v0));
            h1 = next_halfedge(v1v0);
            h2 = next_halfedge(h1);
            if (is_boundary(opposite_halfedge(h1)) &&
                is_boundary(opposite_halfedge(h2)))
                return false;
        }

        // if vl and vr are equal or both invalid -> fail
        if (vl == vr)
            return false;

        // edge between two boundary vertices should be a boundary edge
        if (is_boundary(v0) && is_boundary(v1) && !is_boundary(v0v1) &&
            !is_boundary(v1v0))
            return false;

        // test intersection of the one-rings of v0 and v1
        for (auto vv: get_vertices(v0)) {
            if (vv != v1 && vv != vl && vv != vr)
                if (find_halfedge(vv, v1).is_valid())
                    return false;
        }

        // passed all tests
        return true;
    }

    void HalfedgeMeshInterface::collapse(Halfedge h) {
        Halfedge h0 = h;
        Halfedge h1 = prev_halfedge(h0);
        Halfedge o0 = opposite_halfedge(h0);
        Halfedge o1 = next_halfedge(o0);

        // remove edge
        remove_edge_helper(h0);

        // remove loops
        if (next_halfedge(next_halfedge(h1)) == h1) {
            remove_loop_helper(h1);
        }

        if (next_halfedge(next_halfedge(o1)) == o1) {
            remove_loop_helper(o1);
        }
    }

    bool HalfedgeMeshInterface::is_removal_ok(Edge e) const {
        Halfedge h0 = get_halfedge(e, 0);
        Halfedge h1 = get_halfedge(e, 1);
        Vertex v0 = to_vertex(h0);
        Vertex v1 = to_vertex(h1);
        Face f0 = face(h0);
        Face f1 = face(h1);

        // boundary?
        if (!f0.is_valid() || !f1.is_valid())
            return false;

        // same face?
        if (f0 == f1)
            return false;

        // are the two faces connect through another vertex?
        for (auto v: get_vertices(f0))
            if (v != v0 && v != v1)
                for (auto f: get_faces(v))
                    if (f == f1)
                        return false;

        return true;
    }

    bool HalfedgeMeshInterface::remove_edge(Edge e) {
        if (!is_removal_ok(e))
            return false;

        Halfedge h0 = get_halfedge(e, 0);
        Halfedge h1 = get_halfedge(e, 1);

        Vertex v0 = to_vertex(h0);
        Vertex v1 = to_vertex(h1);

        Face f0 = face(h0);
        Face f1 = face(h1);

        Halfedge h0_prev = prev_halfedge(h0);
        Halfedge h0_next = next_halfedge(h0);
        Halfedge h1_prev = prev_halfedge(h1);
        Halfedge h1_next = next_halfedge(h1);

        // adjust vertex->halfedge
        if (get_halfedge(v0) == h1)
            set_halfedge(v0, h0_next);
        if (get_halfedge(v1) == h0)
            set_halfedge(v1, h1_next);

        // adjust halfedge->face
        for (auto h: get_halfedges(f0))
            set_face(h, f1);

        // adjust halfedge->halfedge
        set_next_halfedge(h1_prev, h0_next);
        set_next_halfedge(h0_prev, h1_next);

        // adjust face->halfedge
        if (get_halfedge(f1) == h1)
            set_halfedge(f1, h1_next);

        // delete face f0 and edge e
        faces.fdeleted[f0] = true;
        ++faces.deleted_faces;
        edges.edeleted[e] = true;
        ++edges.deleted_edges;

        return true;
    }

    void HalfedgeMeshInterface::split(Face f, Vertex v) {
        // Split an arbitrary face into triangles by connecting each vertex of face
        // f to vertex v . Face f will remain valid (it will become one of the
        // triangles). The halfedge handles of the new triangles will point to the
        // old halfedges.

        Halfedge hend = get_halfedge(f);
        Halfedge h = next_halfedge(hend);

        Halfedge hold = new_edge(to_vertex(hend), v);

        set_next_halfedge(hend, hold);
        set_face(hold, f);

        hold = opposite_halfedge(hold);

        while (h != hend) {
            Halfedge hnext = next_halfedge(h);

            Face fnew = new_face();
            set_halfedge(fnew, h);

            Halfedge hnew = new_edge(to_vertex(h), v);

            set_next_halfedge(hnew, hold);
            set_next_halfedge(hold, h);
            set_next_halfedge(h, hnew);

            set_face(hnew, fnew);
            set_face(hold, fnew);
            set_face(h, fnew);

            hold = opposite_halfedge(hnew);

            h = hnext;
        }

        set_next_halfedge(hold, hend);
        set_next_halfedge(next_halfedge(hend), hold);

        set_face(hold, f);

        set_halfedge(v, hold);
    }

    Halfedge HalfedgeMeshInterface::split(Edge e, Vertex v) {
        Halfedge h0 = get_halfedge(e, 0);
        Halfedge o0 = get_halfedge(e, 1);

        Vertex v2 = to_vertex(o0);

        Halfedge e1 = new_edge(v, v2);
        Halfedge t1 = opposite_halfedge(e1);

        Face f0 = face(h0);
        Face f3 = face(o0);

        set_halfedge(v, h0);
        set_vertex(o0, v);

        if (!is_boundary(h0)) {
            Halfedge h1 = next_halfedge(h0);
            Halfedge h2 = next_halfedge(h1);

            Vertex v1 = to_vertex(h1);

            Halfedge e0 = new_edge(v, v1);
            Halfedge t0 = opposite_halfedge(e0);

            Face f1 = new_face();
            set_halfedge(f0, h0);
            set_halfedge(f1, h2);

            set_face(h1, f0);
            set_face(t0, f0);
            set_face(h0, f0);

            set_face(h2, f1);
            set_face(t1, f1);
            set_face(e0, f1);

            set_next_halfedge(h0, h1);
            set_next_halfedge(h1, t0);
            set_next_halfedge(t0, h0);

            set_next_halfedge(e0, h2);
            set_next_halfedge(h2, t1);
            set_next_halfedge(t1, e0);
        } else {
            set_next_halfedge(prev_halfedge(h0), t1);
            set_next_halfedge(t1, h0);
            // halfedge handle of vh already is h0
        }

        if (!is_boundary(o0)) {
            Halfedge o1 = next_halfedge(o0);
            Halfedge o2 = next_halfedge(o1);

            Vertex v3 = to_vertex(o1);

            Halfedge e2 = new_edge(v, v3);
            Halfedge t2 = opposite_halfedge(e2);

            Face f2 = new_face();
            set_halfedge(f2, o1);
            set_halfedge(f3, o0);

            set_face(o1, f2);
            set_face(t2, f2);
            set_face(e1, f2);

            set_face(o2, f3);
            set_face(o0, f3);
            set_face(e2, f3);

            set_next_halfedge(e1, o1);
            set_next_halfedge(o1, t2);
            set_next_halfedge(t2, e1);

            set_next_halfedge(o0, e2);
            set_next_halfedge(e2, o2);
            set_next_halfedge(o2, o0);
        } else {
            set_next_halfedge(e1, next_halfedge(o0));
            set_next_halfedge(o0, e1);
            set_halfedge(v, e1);
        }

        if (get_halfedge(v2) == h0)
            set_halfedge(v2, t1);

        return t1;
    }

    Halfedge HalfedgeMeshInterface::insert_edge(Halfedge h0, Halfedge h1) {
        assert(face(h0) == face(h1));
        assert(face(h0).is_valid());

        Vertex v0 = to_vertex(h0);
        Vertex v1 = to_vertex(h1);

        Halfedge h2 = next_halfedge(h0);
        Halfedge h3 = next_halfedge(h1);

        Halfedge h4 = new_edge(v0, v1);
        Halfedge h5 = opposite_halfedge(h4);

        Face f0 = face(h0);
        Face f1 = new_face();

        set_halfedge(f0, h0);
        set_halfedge(f1, h1);

        set_next_halfedge(h0, h4);
        set_next_halfedge(h4, h3);
        set_face(h4, f0);

        set_next_halfedge(h1, h5);
        set_next_halfedge(h5, h2);
        Halfedge h = h2;
        do {
            set_face(h, f1);
            h = next_halfedge(h);
        } while (h != h2);

        return h4;
    }

    bool HalfedgeMeshInterface::is_flip_ok(Edge e) const {
        // boundary edges cannot be flipped
        if (is_boundary(e))
            return false;

        // check if the flipped edge is already present in the mesh
        Halfedge h0 = get_halfedge(e, 0);
        Halfedge h1 = get_halfedge(e, 1);

        Vertex v0 = to_vertex(next_halfedge(h0));
        Vertex v1 = to_vertex(next_halfedge(h1));

        if (v0 == v1) // this is generally a bad sign !!!
            return false;

        if (find_halfedge(v0, v1).is_valid())
            return false;

        return true;
    }

    void HalfedgeMeshInterface::flip(Edge e) {
        //let's make it sure it is actually checked
        assert(is_flip_ok(e));

        Halfedge a0 = get_halfedge(e, 0);
        Halfedge b0 = get_halfedge(e, 1);

        Halfedge a1 = next_halfedge(a0);
        Halfedge a2 = next_halfedge(a1);

        Halfedge b1 = next_halfedge(b0);
        Halfedge b2 = next_halfedge(b1);

        Vertex va0 = to_vertex(a0);
        Vertex va1 = to_vertex(a1);

        Vertex vb0 = to_vertex(b0);
        Vertex vb1 = to_vertex(b1);

        Face fa = face(a0);
        Face fb = face(b0);

        set_vertex(a0, va1);
        set_vertex(b0, vb1);

        set_next_halfedge(a0, a2);
        set_next_halfedge(a2, b1);
        set_next_halfedge(b1, a0);

        set_next_halfedge(b0, b2);
        set_next_halfedge(b2, a1);
        set_next_halfedge(a1, b0);

        set_face(a1, fb);
        set_face(b1, fa);

        set_halfedge(fa, a0);
        set_halfedge(fb, b0);

        if (get_halfedge(va0) == b0)
            set_halfedge(va0, a1);
        if (get_halfedge(vb0) == a0)
            set_halfedge(vb0, b1);
    }

    size_t HalfedgeMeshInterface::valence(Vertex v) const {
        auto vv = get_vertices(v);
        return std::distance(vv.begin(), vv.end());
    }

    size_t HalfedgeMeshInterface::valence(Face f) const {
        auto vv = get_vertices(f);
        return std::distance(vv.begin(), vv.end());
    }

    void HalfedgeMeshInterface::delete_vertex(Vertex v) {
        if (vertices.vdeleted[v])
            return;

        // collect incident faces
        std::vector<Face> incident_faces;
        incident_faces.reserve(6);

        for (auto f: get_faces(v))
            incident_faces.push_back(f);

        // delete incident faces
        for (auto f: incident_faces)
            delete_face(f);

        // mark v as deleted if not yet done by delete_face()
        if (!vertices.vdeleted[v]) {
            vertices.vdeleted[v] = true;
            vertices.deleted_vertices++;
        }
    }

    void HalfedgeMeshInterface::delete_edge(Edge e) {
        if (edges.edeleted[e])
            return;

        Face f0 = face(get_halfedge(e, 0));
        Face f1 = face(get_halfedge(e, 1));

        if (f0.is_valid())
            delete_face(f0);
        if (f1.is_valid())
            delete_face(f1);
    }

    void HalfedgeMeshInterface::delete_face(Face f) {
        if (faces.fdeleted[f])
            return;

        // mark face deleted
        if (!faces.fdeleted[f]) {
            faces.fdeleted[f] = true;
            faces.deleted_faces++;
        }

        // boundary edges of face f to be deleted
        std::vector<Edge> deletedEdges;
        deletedEdges.reserve(3);

        // vertices of face f for updating their outgoing halfedge
        std::vector<Vertex> vertex_indices;
        vertex_indices.reserve(3);

        // for all halfedges of face f do:
        //   1) invalidate face handle.
        //   2) collect all boundary halfedges, set them deleted
        //   3) store vertex handles
        for (auto hc: get_halfedges(f)) {
            set_face(hc, Face());

            if (is_boundary(opposite_halfedge(hc)))
                deletedEdges.push_back(edge(hc));

            vertex_indices.push_back(to_vertex(hc));
        }

        // delete all collected (half)edges
        // delete isolated vertices
        if (!deletedEdges.empty()) {
            auto delit(deletedEdges.begin()), delend(deletedEdges.end());

            Halfedge h0, h1, next0, next1, prev0, prev1;
            Vertex v0, v1;

            for (; delit != delend; ++delit) {
                h0 = get_halfedge(*delit, 0);
                v0 = to_vertex(h0);
                next0 = next_halfedge(h0);
                prev0 = prev_halfedge(h0);

                h1 = get_halfedge(*delit, 1);
                v1 = to_vertex(h1);
                next1 = next_halfedge(h1);
                prev1 = prev_halfedge(h1);

                // adjust next and prev handles
                set_next_halfedge(prev0, next1);
                set_next_halfedge(prev1, next0);

                // mark edge deleted
                if (!edges.edeleted[*delit]) {
                    edges.edeleted[*delit] = true;
                    edges.deleted_edges++;
                }

                // update v0
                if (get_halfedge(v0) == h1) {
                    if (next0 == h1) {
                        if (!vertices.vdeleted[v0]) {
                            vertices.vdeleted[v0] = true;
                            vertices.deleted_vertices++;
                        }
                    } else
                        set_halfedge(v0, next0);
                }

                // update v1
                if (get_halfedge(v1) == h0) {
                    if (next1 == h0) {
                        if (!vertices.vdeleted[v1]) {
                            vertices.vdeleted[v1] = true;
                            vertices.deleted_vertices++;
                        }
                    } else
                        set_halfedge(v1, next1);
                }
            }
        }

        // update outgoing halfedge handles of remaining vertices
        auto vit(vertices.begin()), vend(vertices.end());
        for (; vit != vend; ++vit)
            adjust_outgoing_halfedge(*vit);
    }

    Halfedge HalfedgeMeshInterface::new_edge() {
        edges.push_back();
        halfedges.push_back();
        halfedges.push_back();

        Halfedge h0(static_cast<IndexType>(halfedges.size()) - 2);
        Halfedge h1(static_cast<IndexType>(halfedges.size()) - 1);

        return h0;
    }

    Halfedge HalfedgeMeshInterface::new_edge(Vertex start, Vertex end) {
        assert(start != end);

        edges.push_back();
        halfedges.push_back();
        halfedges.push_back();

        Halfedge h0(static_cast<IndexType>(halfedges.size()) - 2);
        Halfedge h1(static_cast<IndexType>(halfedges.size()) - 1);

        set_vertex(h0, end);
        set_vertex(h1, start);

        return h0;
    }

    void HalfedgeMeshInterface::adjust_outgoing_halfedge(Vertex v) {
        Halfedge h = get_halfedge(v);
        const Halfedge hh = h;

        if (h.is_valid()) {
            do {
                if (is_boundary(h)) {
                    set_halfedge(v, h);
                    return;
                }
                h = rotate_cw(h);
            } while (h != hh);
        }
    }

    void HalfedgeMeshInterface::remove_edge_helper(Halfedge h) {
        Halfedge hn = next_halfedge(h);
        Halfedge hp = prev_halfedge(h);

        Halfedge o = opposite_halfedge(h);
        Halfedge on = next_halfedge(o);
        Halfedge op = prev_halfedge(o);

        Face fh = face(h);
        Face fo = face(o);

        Vertex vh = to_vertex(h);
        Vertex vo = to_vertex(o);

        // halfedge -> vertex
        for (const auto hc: halfedges(vo)) {
            set_vertex(opposite_halfedge(hc), vh);
        }

        // halfedge -> halfedge
        set_next_halfedge(hp, hn);
        set_next_halfedge(op, on);

        // face -> halfedge
        if (fh.is_valid())
            set_halfedge(fh, hn);
        if (fo.is_valid())
            set_halfedge(fo, on);

        // vertex -> halfedge
        if (get_halfedge(vh) == o)
            set_halfedge(vh, hn);
        adjust_outgoing_halfedge(vh);
        set_halfedge(vo, Halfedge());

        // delete stuff
        vertices.vdeleted[vo] = true;
        ++vertices.deleted_vertices;
        edges.edeleted[edge(h)] = true;
        ++edges.deleted_edges;
    }

    void HalfedgeMeshInterface::remove_loop_helper(Halfedge h) {
        Halfedge h0 = h;
        Halfedge h1 = next_halfedge(h0);

        Halfedge o0 = opposite_halfedge(h0);
        Halfedge o1 = opposite_halfedge(h1);

        Vertex v0 = to_vertex(h0);
        Vertex v1 = to_vertex(h1);

        Face fh = face(h0);
        Face fo = face(o0);

        // is it a loop ?
        assert((next_halfedge(h1) == h0) && (h1 != o0));

        // halfedge -> halfedge
        set_next_halfedge(h1, next_halfedge(o0));
        set_next_halfedge(prev_halfedge(o0), h1);

        // halfedge -> face
        set_face(h1, fo);

        // vertex -> halfedge
        set_halfedge(v0, h1);
        adjust_outgoing_halfedge(v0);
        set_halfedge(v1, o1);
        adjust_outgoing_halfedge(v1);

        // face -> halfedge
        if (fo.is_valid() && get_halfedge(fo) == o0)
            set_halfedge(fo, h1);

        // delete stuff
        if (fh.is_valid()) {
            faces.fdeleted[fh] = true;
            ++faces.deleted_faces;
        }
        edges.edeleted[edge(h)] = true;
        ++edges.deleted_edges;
    }
}