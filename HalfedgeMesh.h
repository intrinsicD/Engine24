//
// Created by alex on 24.06.24.
//

#ifndef ENGINE24_HALFEDGEMESH_H
#define ENGINE24_HALFEDGEMESH_H

#include <vector>
#include <unordered_map>

namespace Bcg {
    struct HalfedgeMesh {
        struct VertexConnectivity {
            unsigned int h = -1;
        };
        struct HalfedgeConnectivity {
            unsigned int v = -1;
            unsigned int nh = -1;
            unsigned int ph = -1;
            unsigned int f = -1;
        };
        struct FaceConnectivity {
            unsigned int h = -1;
        };
        std::vector<VertexConnectivity> vertices;
        std::vector<HalfedgeConnectivity> halfedges;
        std::vector<FaceConnectivity> faces;

        std::vector<bool> add_face_is_new_;
        std::vector<unsigned int> add_face_halfedges_;
        std::vector<bool> add_face_needs_adjust_;
        using NextCacheEntry = std::pair<unsigned int, unsigned int>;
        using NextCache = std::vector<NextCacheEntry>;
        NextCache add_face_next_cache_;

        inline bool is_valid(unsigned int index) const {
            return index != -1;
        }

        inline unsigned int get_opposite(unsigned int h) const {
            return ((h & 1) ? h - 1 : h + 1);
        }

        inline unsigned int ccw_rotated_halfedge(unsigned int h) const {
            return get_opposite(halfedges[h].ph);
        }

        inline unsigned int cw_rotated_halfedge(unsigned int h) const {
            return halfedges[get_opposite(h)].nh;
        }

        inline unsigned int new_edge(unsigned int start, unsigned int end) {
            halfedges.push_back({});
            halfedges.push_back({});

            unsigned int h = halfedges.size() - 2;
            unsigned int oh = halfedges.size() - 1;
            halfedges[h].v = start;
            halfedges[oh].v = end;
            return h;
        }

        unsigned int find_halfedge(unsigned int start, unsigned int end) const {
            assert(is_valid(start) && is_valid(end));

            unsigned int h = vertices[start].h;
            const unsigned int hh = h;

            if (is_valid(h)) {
                do {
                    if (halfedges[h].v == end)
                        return h;
                    h = cw_rotated_halfedge(h);
                } while (h != hh);
            }

            return -1;
        }

        inline bool is_vertex_boundary(unsigned int v) const {
            unsigned int h = vertices[v].h;
            return (!(is_valid(h) && is_valid(halfedges[h].f)));
        }

        inline bool is_halfedge_boundary(unsigned int h) const {
            return (!is_valid(halfedges[h].f));
        }

        bool is_face_boundary(unsigned int f) const {
            unsigned int h = faces[f].h;
            const unsigned int hh = h;
            do {
                if (is_halfedge_boundary(get_opposite(h))) {
                    return true;
                }
                h = halfedges[h].nh;
            } while (h != hh);
            return false;
        }

        void adjust_outgoing_halfedge(unsigned v) {
            unsigned int h = vertices[v].h;
            const unsigned int hh = h;

            if (is_valid(h)) {
                do {
                    if (is_halfedge_boundary(h)) {
                        vertices[v].h = h;
                        return;
                    }
                    h = cw_rotated_halfedge(h);
                } while (h != hh);
            }
        }

        inline unsigned int add_triangle(unsigned int v0, unsigned int v1, unsigned int v2) {
            return add_face({v0, v1, v2});
        }

        inline unsigned int add_face(std::vector<unsigned int> face) {
            const size_t n(face.size());
            std::vector<bool> &is_new = add_face_is_new_;
            std::vector<unsigned int> &face_halfedges = add_face_halfedges_;
            std::vector<bool> &needs_adjust = add_face_needs_adjust_;
            NextCache &next_cache = add_face_next_cache_;
            is_new.clear();
            is_new.resize(n);
            face_halfedges.clear();
            face_halfedges.resize(n);
            needs_adjust.clear();
            needs_adjust.resize(n, false);
            next_cache.clear();
            next_cache.reserve(3 * n);
// test for topological errors
            unsigned int i, ii, id;
            unsigned int inner_next, inner_prev, outer_next, outer_prev, boundary_next,
                    boundary_prev, patch_start, patch_end;
            for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n) {
                if (!is_vertex_boundary(face[i])) {
                    auto what = "SurfaceMesh::add_face: Complex vertex.";
                    throw std::logic_error(what);
                }

                face_halfedges[i] = find_halfedge(face[i], face[ii]);
                is_new[i] = !is_valid(face_halfedges[i]);

                if (!is_new[i] && !is_halfedge_boundary(face_halfedges[i])) {
                    auto what = "SurfaceMesh::add_face: Complex edge.";
                    throw std::logic_error(what);
                }
            }
            // re-link patches if necessary
            for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n) {
                if (!is_new[i] && !is_new[ii]) {
                    inner_prev = face_halfedges[i];
                    inner_next = face_halfedges[ii];

                    if (halfedges[inner_prev].nh != inner_next) {
                        // here comes the ugly part... we have to relink a whole patch

                        // search a free gap
                        // free gap will be between boundaryPrev and boundaryNext
                        outer_prev = get_opposite(inner_next);
                        outer_next = get_opposite(inner_prev);
                        boundary_prev = outer_prev;
                        do {
                            boundary_prev = get_opposite(halfedges[boundary_prev].nh);
                        } while (!is_halfedge_boundary(boundary_prev) || boundary_prev == inner_prev);
                        boundary_next = halfedges[boundary_prev].nh;
                        assert(is_halfedge_boundary(boundary_prev));
                        assert(is_halfedge_boundary(boundary_next));

                        // ok ?
                        if (boundary_next == inner_next) {
                            auto what =
                                    "SurfaceMesh::add_face: Patch re-linking failed.";
                            throw std::logic_error(what);
                        }

                        // other halfedges' handles
                        patch_start = halfedges[inner_prev].nh;
                        patch_end = halfedges[inner_next].ph;

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
                    face_halfedges[i] = new_edge(face[i], face[ii]);
                }
            }

            // create the face
            faces.push_back({});
            unsigned int f = faces.size() - 1;
            faces[f].h = face_halfedges[n - 1];

            // setup halfedges
            unsigned int v;
            for (i = 0, ii = 1; i < n; ++i, ++ii, ii %= n) {
                v = face[ii];
                inner_prev = face_halfedges[i];
                inner_next = face_halfedges[ii];

                id = (is_new[i] ? 1 : 0) | (is_new[ii] ? 2 : 0);

                if (id) {
                    outer_prev = get_opposite(inner_next);
                    outer_next = get_opposite(inner_prev);

                    // set outer links
                    switch (id) {
                        case 1: // prev is new, next is old
                            boundary_prev = halfedges[inner_next].ph;
                            next_cache.emplace_back(boundary_prev, outer_next);
                            vertices[v].h = outer_next;
                            break;

                        case 2: // next is new, prev is old
                            boundary_next = halfedges[inner_prev].nh;
                            next_cache.emplace_back(outer_prev, boundary_next);
                            vertices[v].h = boundary_next;
                            break;

                        case 3: // both are new
                            if (!is_valid(vertices[v].h)) {
                                vertices[v].h = outer_next;
                                next_cache.emplace_back(outer_prev, outer_next);
                            } else {
                                boundary_next = vertices[v].h;
                                boundary_prev = halfedges[boundary_next].ph;
                                next_cache.emplace_back(boundary_prev, outer_next);
                                next_cache.emplace_back(outer_prev, boundary_next);
                            }
                            break;
                    }

                    // set inner link
                    next_cache.emplace_back(inner_prev, inner_next);
                } else
                    needs_adjust[ii] = (vertices[v].h == inner_next);

                // set face handle
                halfedges[face_halfedges[i]].f = f;
            }

            // process next halfedge cache
            for (const auto& [prev, next] : next_cache) {
                halfedges[prev].nh = next;
                halfedges[next].ph = prev;
            }

            // adjust vertices' halfedge handle
            for (i = 0; i < n; ++i) {
                if (needs_adjust[i]) {
                    adjust_outgoing_halfedge(face[i]);
                }
            }

            return f;
        }

        void build(unsigned int num_vertices, const std::vector<unsigned int> &triangles) {
            vertices.clear();
            halfedges.clear();
            faces.clear();

            unsigned int num_triangles = triangles.size() / 3;
            vertices.reserve(num_vertices);
            halfedges.reserve(6 * num_triangles);
            faces.reserve(num_triangles);

            Log::Info("Process vertices");
            for (unsigned int i = 0; i < num_vertices; ++i) {
                vertices.emplace_back();
                vertices.back().h = -1;
            }
            Log::Info("Process triangles");
            for (unsigned int i = 0; i < num_triangles; ++i) {
                unsigned int face_index = 3 * i;
                unsigned int f = add_face(
                        {triangles[face_index], triangles[face_index + 1], triangles[face_index + 2]});
            }
        }
    };
}

#endif //ENGINE24_HALFEDGEMESH_H
