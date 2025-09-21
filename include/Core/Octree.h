#pragma once

#include "Properties.h"
#include "GeometricContainment.h"
#include "GeometricIntersection.h"
#include "Logger.h"

#include <numeric>
#include <queue>
#include <array>
#include <algorithm>
#include <limits>
#include <iterator>

namespace Bcg {
    class Octree {
    public:
        struct Node {
            AABB<float> aabb;
            size_t first_element = std::numeric_limits<size_t>::max();
            size_t size = std::numeric_limits<size_t>::max();
            std::array<size_t, 8> children;
            bool is_leaf = true;

            Node() {
                children.fill(std::numeric_limits<size_t>::max());
            }

            friend std::ostream &operator<<(std::ostream &os, const Node &n) {
                os << " aabb_ min: " << n.aabb.min;
                os << " aabb_ max: " << n.aabb.max;
                os << " first_element: " << n.first_element;
                os << " size: " << n.size;
                os << " is_leaf: " << n.is_leaf;
                os << " children: ";
                for (const auto &c: n.children) {
                    os << c << " ";
                }
                return os;
            }
        };

        enum class SplitPoint { Center, Mean, Median };

        struct SplitPolicy {
            SplitPoint split_point = SplitPoint::Center;
            bool tight_children = false; // shrink child boxes to exactly fit contents
            float epsilon = 0.0f; // optional padding when tightening
        };


        PropertyContainer octree;
        Property<Node> nodes;

        Property<AABB<float> > element_aabbs;

        size_t get_max_elements_per_node() const noexcept {
            return max_elements_per_node;
        }

        size_t get_max_octree_depth() const noexcept {
            return max_octree_depth;
        }

        const SplitPolicy &get_split_policy() const noexcept {
            return split_policy;
        }

        const std::vector<size_t> &get_element_indices() const noexcept {
            return element_indices;
        }

        void build(const Property<AABB<float> > &aabbs, const SplitPolicy &policy, const size_t max_per_node,
                   const size_t max_depth) {
            element_aabbs = aabbs;

            if (!element_aabbs) {
                Log::Error("Element AABBs property is not set. Cannot build octree.");
                return;
            }

            split_policy = policy;
            max_elements_per_node = max_per_node;
            max_octree_depth = max_depth;

            octree.clear(); // Clear previous state
            const size_t num_elements = element_aabbs.vector().size();

            if (num_elements == 0) {
                element_indices.clear();
                return;
            }

            element_indices.resize(num_elements);
            std::iota(element_indices.begin(), element_indices.end(), 0);

            nodes = octree.add<Node>("n:nodes");

            // Create root node
            const size_t root_idx = create_node();
            nodes[root_idx].first_element = 0;
            nodes[root_idx].size = num_elements;
            nodes[root_idx].aabb = AABB<float>::Build(element_aabbs.vector().begin(), element_aabbs.vector().end());

            subdivide(root_idx, 0);
        }

        void query(const AABB<float> &query_aabb, std::vector<size_t> &result) const {
            result.clear();
            if (octree.size() > 0) {
                query_recursive(0, query_aabb, result);
            }
        }

        void query(const Sphere<float> &query_sphere, std::vector<size_t> &result) const {
            result.clear();
            if (octree.size() > 0) {
                query_recursive(0, query_sphere, result);
            }
        }

        using QueueElement = std::pair<float, size_t>;

        void query_knn(const Vector<float, 3> &query_point, size_t k, std::vector<size_t> &result) const {
            result.clear();
            if (octree.size() == 0 || k == 0) {
                return;
            }

            std::priority_queue<QueueElement> pq;

            query_knn_recursive(0, query_point, k, pq);

            result.resize(pq.size());
            size_t i = pq.size();
            while (!pq.empty()) {
                result[--i] = pq.top().second;
                pq.pop();
            }
        }

        void query_nearest(const Vector<float, 3> &query_point, size_t &result) const {
            result = std::numeric_limits<size_t>::max();
            if (octree.size() > 0) {
                float min_dist_sq = std::numeric_limits<float>::max();
                query_recursive(0, query_point, result, min_dist_sq);
            }
        }

    private:
        size_t create_node() {
            octree.push_back();
            return octree.size() - 1;
        }

        void subdivide(const size_t node_idx, size_t depth) {
            Node &node = nodes[node_idx];

            if (depth >= max_octree_depth || nodes[node_idx].size <= max_elements_per_node) {
                nodes[node_idx].is_leaf = true;
                return;
            }

            Vector<float, 3> sp = choose_split_point(node_idx);

            // A predicate to determine which child an element belongs to.
            auto get_child_idx = [&](size_t elem_idx) {
                const auto elem_center = element_aabbs[elem_idx].center();
                int child_idx = 0;
                if (elem_center[0] >= sp[0]) child_idx |= 1;
                if (elem_center[1] >= sp[1]) child_idx |= 2;
                if (elem_center[2] >= sp[2]) child_idx |= 4;
                return child_idx;
            };

            size_t start = nodes[node_idx].first_element;
            size_t end = nodes[node_idx].first_element + nodes[node_idx].size;

            // 1. Count elements per child
            std::array<size_t, 8> child_sizes{};
            for (size_t i = 0; i < nodes[node_idx].size; ++i) {
                size_t elem_idx = element_indices[start + i];
                child_sizes[get_child_idx(elem_idx)]++;
            }

            // All points in one child?
            size_t non_empty = 0;
            for (int i = 0; i < 8; ++i) {
                if (child_sizes[i] > 0) {
                    ++non_empty;
                }
            }
            // If all elements in exactly one child, bail out to leaf
            if (non_empty <= 1) {
                nodes[node_idx].is_leaf = true;
                return;
            }

            // 2. Determine offsets
            std::array<size_t, 8> child_offsets{};
            child_offsets[0] = start;
            for (int i = 1; i < 8; ++i) {
                child_offsets[i] = child_offsets[i - 1] + child_sizes[i - 1];
            }

            // 3. Re-arrange elements in-place (single pass)

            scratch_indices.clear();
            scratch_indices.reserve(nodes[node_idx].size);
            std::copy(element_indices.begin() + start, element_indices.begin() + end,
                      std::back_inserter(scratch_indices));

            std::array<size_t, 8> cur = child_offsets;
            for (size_t elem_idx: scratch_indices) {
                int child_idx = get_child_idx(elem_idx);
                element_indices[cur[child_idx]++] = elem_idx;
            }

            // Create child nodes using the calculated offsets and sizes
            nodes[node_idx].is_leaf = false;
            for (int i = 0; i < 8; ++i) {
                if (child_sizes[i] > 0) {
                    const size_t child_node_idx = create_node();
                    nodes[node_idx].children[i] = child_node_idx;

                    Node &child = nodes[child_node_idx];
                    child.first_element = child_offsets[i];
                    child.size = child_sizes[i];
                    const auto last_element = child.first_element + child.size;

                    if (split_policy.tight_children) {
                        // This helper function will need to be updated (see below)
                        nodes[child_node_idx].aabb = tight_child_aabb(
                            element_indices.begin() + child.first_element,
                            element_indices.begin() + last_element,
                            split_policy.epsilon
                        );
                    } else {
                        // This logic for non-tight AABBs remains the same
                        const auto min_p = nodes[node_idx].aabb.min;
                        const auto max_p = nodes[node_idx].aabb.max;
                        nodes[child_node_idx].aabb.min = {
                            (i & 1) ? sp[0] : min_p[0], (i & 2) ? sp[1] : min_p[1], (i & 4) ? sp[2] : min_p[2]
                        };
                        nodes[child_node_idx].aabb.max = {
                            (i & 1) ? max_p[0] : sp[0], (i & 2) ? max_p[1] : sp[1], (i & 4) ? max_p[2] : sp[2]
                        };
                    }

                    subdivide(child_node_idx, depth + 1);
                }
            }
        }

        void query_recursive(const size_t node_idx, const AABB<float> &query_aabb,
                             std::vector<size_t> &result) const {
            const Node &node = nodes[node_idx];
            const AABB<float> &nb = nodes[node_idx].aabb;

            if (!IntersectsTraits<AABB<float>, AABB<float> >::intersects(nb, query_aabb)) {
                return;
            }

            if (ContainsTraits<AABB<float>, AABB<float> >::contains(query_aabb, nb)) {
                // Query fully contains the node: all elements intersect the query AABB.
                for (size_t i = 0; i < node.size; ++i) {
                    size_t elem_idx = element_indices[node.first_element + i];
                    result.push_back(elem_idx);
                }
                return;
            }

            if (node.is_leaf) {
                for (size_t i = 0; i < node.size; ++i) {
                    size_t elem_idx = element_indices[node.first_element + i];
                    if (IntersectsTraits<AABB<float>, AABB<float> >::intersects(element_aabbs[elem_idx], query_aabb)) {
                        result.push_back(elem_idx);
                    }
                }
            } else {
                for (const size_t child_idx: node.children) {
                    if (child_idx != std::numeric_limits<size_t>::max()) {
                        query_recursive(child_idx, query_aabb, result);
                    }
                }
            }
        }

        void query_recursive(size_t node_idx, const Sphere<float> &query_sphere,
                             std::vector<size_t> &result) const {
            const Node &node = nodes[node_idx];
            const AABB<float> &nb = nodes[node_idx].aabb;

            // prune if no overlap using Minkowski sum logic
            if (!IntersectsTraits<Sphere<float>, AABB<float> >::intersects(query_sphere, nb)) {
                return;
            }

            if (ContainsTraits<Sphere<float>, AABB<float> >::contains(query_sphere, nb)) {
                // Sphere fully contains the node's AABB => every element in this node intersects the sphere.
                for (size_t i = 0; i < node.size; ++i) {
                    size_t elem_idx = element_indices[node.first_element + i];
                    result.push_back(elem_idx);
                }
                return; // early pruning
            }

            if (node.is_leaf) {
                for (size_t i = 0; i < node.size; ++i) {
                    size_t elem_idx = element_indices[node.first_element + i];
                    if (IntersectsTraits<Sphere<float>, AABB<
                        float> >::intersects(query_sphere, element_aabbs[elem_idx])) {
                        result.push_back(elem_idx);
                    }
                }
            } else {
                for (size_t child_idx: node.children) {
                    if (child_idx != std::numeric_limits<size_t>::max()) {
                        query_recursive(child_idx, query_sphere, result);
                    }
                }
            }
        }

        void query_knn_recursive(size_t node_idx, const Vector<float, 3> &query_point, size_t k,
                                 std::priority_queue<QueueElement> &pq) const {
            const Node &node = nodes[node_idx];

            if (node.is_leaf) {
                for (size_t i = 0; i < node.size; ++i) {
                    const size_t elem_idx = element_indices[node.first_element + i];
                    const float dist_sq = SquaredDistanceTraits<AABB<float>, Vector<float, 3> >::squared_distance(
                        element_aabbs[elem_idx], query_point);

                    if (pq.size() < k) {
                        pq.push({dist_sq, elem_idx});
                    } else if (dist_sq < pq.top().first) {
                        pq.pop();
                        pq.push({dist_sq, elem_idx});
                    }
                }
            } else {
                std::array<std::pair<float, size_t>, 8> child_distances;
                size_t num_children = 0;
                for (size_t child_idx: node.children) {
                    if (child_idx != std::numeric_limits<size_t>::max()) {
                        const auto &child_aabb = nodes[child_idx].aabb;
                        float dist_sq = SquaredDistanceTraits<AABB<float>, Vector<float,
                            3> >::squared_distance(child_aabb, query_point);
                        child_distances[num_children++] = {dist_sq, child_idx};
                    }
                }

                // Sort only the valid children
                std::sort(child_distances.begin(), child_distances.begin() + num_children);
                float thr = (pq.size() == k) ? pq.top().first : std::numeric_limits<float>::infinity();

                for (size_t i = 0; i < num_children; ++i) {
                    if (child_distances[i].first > thr) break;
                    query_knn_recursive(child_distances[i].second, query_point, k, pq);
                }
            }
        }

        void query_recursive(size_t node_idx, const Vector<float, 3> &query_point,
                             size_t &result, float &min_dist_sq) const {
            const Node &node = nodes[node_idx];
            const AABB<float> &node_aabb = nodes[node_idx].aabb;

            Vector<float, 3> closest_point = ClosestPointTraits<AABB<float>, Vector<float,
                3> >::closest_point(node_aabb, query_point);

            float dist_sq_to_box = VecTraits<Vector<float, 3> >::squared_length(
                closest_point - query_point);
            if (dist_sq_to_box >= min_dist_sq) {
                return;
            }

            if (node.is_leaf) {
                for (size_t i = 0; i < node.size; ++i) {
                    size_t elem_idx = element_indices[node.first_element + i];
                    const float dist_sq = SquaredDistanceTraits<AABB<float>, Vector<float, 3> >::squared_distance(
                        element_aabbs[elem_idx], query_point);
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        result = elem_idx;
                    }
                }
            } else {
                std::array<std::pair<float, size_t>, 8> child_distances;
                size_t num_children = 0;
                for (size_t child_idx: node.children) {
                    if (child_idx != std::numeric_limits<size_t>::max()) {
                        const auto &child_aabb = nodes[child_idx].aabb;
                        Vector<float, 3> closest_point_child = ClosestPointTraits<AABB<float>, Vector<float,
                            3> >::closest_point(child_aabb, query_point);
                        float dist_sq = VecTraits<Vector<float, 3> >::squared_length(
                            closest_point_child - query_point);
                        child_distances[num_children++] = {dist_sq, child_idx};
                    }
                }

                // Sort only the valid children
                std::sort(child_distances.begin(), child_distances.begin() + num_children);

                for (size_t i = 0; i < num_children; ++i) {
                    // Prune and recurse
                    if (child_distances[i].first >= min_dist_sq) break;
                    query_recursive(child_distances[i].second, query_point, result, min_dist_sq);
                }
            }
        }

        Vector<float, 3> compute_mean_center(size_t first, size_t size, const Vector<float, 3> &fallback_center) const {
            if (size == 0) {
                return fallback_center; // fallback; or pass node_idx and use aabbs[node_idx]
            }
            Vector<float, 3> acc(0.0f, 0.0f, 0.0f);
            for (size_t i = 0; i < size; ++i) {
                const auto idx = element_indices[first + i];
                acc += element_aabbs[idx].center();
            }
            return acc / float(size);
        }

        Vector<float, 3> compute_median_center(size_t first, size_t size,
                                               const Vector<float, 3> &fallback_center) const {
            if (size == 0) {
                return fallback_center; // fallback; or pass node_idx and use aabbs[node_idx]
            }
            std::vector<Vector<float, 3> > centers;
            centers.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                centers.push_back(element_aabbs[element_indices[first + i]].center());
            }
            const size_t median_idx = centers.size() / 2;
            auto kth = [](std::vector<Vector<float, 3> > &centers, size_t median_idx, int dim) {
                std::nth_element(centers.begin(), centers.begin() + median_idx, centers.end(),
                                 [dim](const auto &a, const auto &b) { return a[dim] < b[dim]; });
                return centers[median_idx][dim];
            };
            return {kth(centers, median_idx, 0), kth(centers, median_idx, 1), kth(centers, median_idx, 2)};
        }

        Vector<float, 3> choose_split_point(size_t node_idx) const {
            const auto &node = nodes[node_idx];
            const Vector<float, 3> fallback_center = nodes[node_idx].aabb.center();
            switch (split_policy.split_point) {
                case SplitPoint::Mean: return compute_mean_center(node.first_element, node.size, fallback_center);
                case SplitPoint::Median: return compute_median_center(node.first_element, node.size, fallback_center);
                case SplitPoint::Center:
                default: return fallback_center;
            }
        }

        template<typename FwdIt>
        AABB<float> tight_child_aabb(FwdIt begin, FwdIt end, float eps = 0.0f) const {
            if (begin == end) {
                return AABB<float>(); // Return an explicitly invalid AABB
            }

            AABB<float> tight = element_aabbs[*begin];

            for (auto it = std::next(begin); it != end; ++it) {
                tight.merge(element_aabbs[*it]);
            }

            if (eps > 0.0f) {
                Vector<float, 3> padding(eps, eps, eps);
                tight.min -= padding;
                tight.max += padding;
            }
            return tight;
        }

        AABB<float> tight_child_aabb(const std::vector<size_t> &elems, float eps = 0.0f) const {
            return tight_child_aabb(elems.begin(), elems.end(), eps);
        }

        size_t max_elements_per_node = 10;
        size_t max_octree_depth = 10;
        SplitPolicy split_policy;
        std::vector<size_t> element_indices;
        std::vector<size_t> scratch_indices;
    };
}
