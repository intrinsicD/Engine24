#pragma once

#include "Properties.h"
#include "GeometricContainment.h"
#include "GeometricIntersection.h"
#include "Logger.h"
#include "BoundedHeap.h"

#include <numeric>
#include <array>
#include <algorithm>
#include <queue>
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

        [[nodiscard]] size_t get_max_elements_per_node() const noexcept {
            return max_elements_per_node;
        }

        [[nodiscard]] size_t get_max_octree_depth() const noexcept {
            return max_octree_depth;
        }

        [[nodiscard]] const SplitPolicy &get_split_policy() const noexcept {
            return split_policy;
        }

        [[nodiscard]] const std::vector<size_t> &get_element_indices() const noexcept {
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

            subdivide_volume(root_idx, 0);
        }

        void query(const AABB<float> &query_aabb, std::vector<size_t> &result) const {
            result.clear();
            if (octree.empty()) return;

            std::vector<size_t> stack;
            stack.push_back(0); // Start with the root node

            while (!stack.empty()) {
                const size_t node_idx = stack.back();
                stack.pop_back();

                const Node &node = nodes[node_idx];

                // Check elements stored at the current node
                for (size_t i = 0; i < node.size; ++i) {
                    size_t elem_idx = element_indices[node.first_element + i];
                    if (IntersectsTraits<AABB<float>, AABB<float> >::intersects(element_aabbs[elem_idx], query_aabb)) {
                        result.push_back(elem_idx);
                    }
                }

                // Add intersecting child nodes to the stack for future processing
                if (!node.is_leaf) {
                    for (const size_t child_idx: node.children) {
                        if (child_idx != std::numeric_limits<size_t>::max() &&
                            IntersectsTraits<AABB<float>, AABB<
                                float> >::intersects(nodes[child_idx].aabb, query_aabb)) {
                            stack.push_back(child_idx);
                        }
                    }
                }
            }
        }

        void query(const Sphere<float> &query_sphere, std::vector<size_t> &result) const {
            result.clear();
            if (octree.empty()) return;

            std::vector<size_t> stack;
            stack.push_back(0); // Start with the root node

            while (!stack.empty()) {
                const size_t node_idx = stack.back();
                stack.pop_back();

                const Node &node = nodes[node_idx];

                // Check elements stored at the current node
                for (size_t i = 0; i < node.size; ++i) {
                    size_t elem_idx = element_indices[node.first_element + i];
                    if (IntersectsTraits<AABB<float>, Sphere<
                        float> >::intersects(element_aabbs[elem_idx], query_sphere)) {
                        result.push_back(elem_idx);
                    }
                }

                // Add intersecting child nodes to the stack for future processing
                if (!node.is_leaf) {
                    for (const size_t child_idx: node.children) {
                        if (child_idx != std::numeric_limits<size_t>::max() &&
                            IntersectsTraits<AABB<float>, Sphere<float> >::intersects(
                                nodes[child_idx].aabb, query_sphere)) {
                            stack.push_back(child_idx);
                        }
                    }
                }
            }
        }

        using QueueElement = std::pair<float, size_t>;

        void query_knn(const Vector<float, 3> &query_point, size_t k, std::vector<size_t> &result) const {
            result.clear();
            if (octree.empty() || k == 0) {
                return;
            }

            BoundedHeap<QueueElement> bounded_heap(k);

            // The traversal priority queue: stores {distance, node_idx}.
            // We use std::greater to make it a min-priority queue.
            using TraversalElement = std::pair<float, size_t>;
            std::priority_queue<TraversalElement, std::vector<TraversalElement>, std::greater<TraversalElement> > pq;

            // Start traversal at the root.
            const size_t root_idx = 0;
            const float root_dist_sq = SquaredDistanceTraits<AABB<float>, Vector<float, 3> >::squared_distance(
                nodes[root_idx].aabb, query_point);
            pq.push({root_dist_sq, root_idx});

            while (!pq.empty()) {
                const float node_dist_sq = pq.top().first;
                const size_t node_idx = pq.top().second;
                pq.pop();

                // --- MASTER PRUNING STEP ---
                // If the closest node in our queue is already farther than our k-th candidate,
                // then no node or element we could possibly visit can be a better candidate.
                // We can terminate the entire search.
                if (bounded_heap.size() == k && node_dist_sq > bounded_heap.top().first) {
                    break;
                }

                const Node &node = nodes[node_idx];

                // --- Step 1: Process elements residing in THIS node ---
                for (size_t i = 0; i < node.size; ++i) {
                    const size_t elem_idx = element_indices[node.first_element + i];
                    const float dist_sq = SquaredDistanceTraits<AABB<float>, Vector<float, 3> >::squared_distance(
                        element_aabbs[elem_idx], query_point);
                    bounded_heap.push({dist_sq, elem_idx});
                }

                // --- Step 2: Add promising children to the queue ---
                if (!node.is_leaf) {
                    for (const size_t child_idx: node.children) {
                        if (child_idx != std::numeric_limits<size_t>::max()) {
                            const float child_dist_sq = SquaredDistanceTraits<AABB<float>, Vector<float,
                                3> >::squared_distance(
                                nodes[child_idx].aabb, query_point);

                            // We only need to add a child if it could possibly contain a better point.
                            if (bounded_heap.size() < k || child_dist_sq < bounded_heap.top().first) {
                                pq.push({child_dist_sq, child_idx});
                            }
                        }
                    }
                }
            }

            // Extract the results from the heap
            auto sorted_pairs = bounded_heap.get_sorted_data();
            result.resize(sorted_pairs.size());
            for (size_t i = 0; i < sorted_pairs.size(); ++i) {
                result[i] = sorted_pairs[i].second;
            }
        }

        void query_nearest(const Vector<float, 3> &query_point, size_t &result) const {
            result = std::numeric_limits<size_t>::max();
            if (octree.empty()) {
                return;
            }

            float min_dist_sq = std::numeric_limits<float>::max();

            // The traversal priority queue: stores {distance, node_idx}.
            // We use std::greater to make it a min-priority queue.
            using TraversalElement = std::pair<float, size_t>;
            std::priority_queue<TraversalElement, std::vector<TraversalElement>, std::greater<TraversalElement> > pq;

            // Start traversal at the root.
            const size_t root_idx = 0;
            const float root_dist_sq = SquaredDistanceTraits<AABB<float>, Vector<float, 3> >::squared_distance(
                nodes[root_idx].aabb, query_point);
            pq.push({root_dist_sq, root_idx});

            while (!pq.empty()) {
                const float node_dist_sq = pq.top().first;
                const size_t node_idx = pq.top().second;
                pq.pop();

                // --- MASTER PRUNING STEP ---
                // If the closest node in our queue is already farther than our best found point,
                // we can terminate the entire search. Nothing else in the queue can be better.
                if (node_dist_sq >= min_dist_sq) {
                    break;
                }

                const Node &node = nodes[node_idx];

                // --- Step 1: Process elements residing in THIS node ---
                for (size_t i = 0; i < node.size; ++i) {
                    const size_t elem_idx = element_indices[node.first_element + i];
                    const float elem_dist_sq = SquaredDistanceTraits<AABB<float>, Vector<float, 3> >::squared_distance(
                        element_aabbs[elem_idx], query_point);

                    if (elem_dist_sq < min_dist_sq) {
                        min_dist_sq = elem_dist_sq;
                        result = elem_idx;
                    }
                }

                // --- Step 2: Add promising children to the queue ---
                if (!node.is_leaf) {
                    for (const size_t child_idx: node.children) {
                        if (child_idx != std::numeric_limits<size_t>::max()) {
                            const float child_dist_sq = SquaredDistanceTraits<AABB<float>, Vector<float,
                                3> >::squared_distance(
                                nodes[child_idx].aabb, query_point);

                            // Only add a child if it could possibly contain a better point.
                            if (child_dist_sq < min_dist_sq) {
                                pq.push({child_dist_sq, child_idx});
                            }
                        }
                    }
                }
            }
        }

    private:
        size_t create_node() {
            octree.push_back();
            return octree.size() - 1;
        }

        void subdivide_volume(const size_t node_idx, size_t depth) {
            if (depth >= max_octree_depth || nodes[node_idx].size <= max_elements_per_node) {
                nodes[node_idx].is_leaf = true;
                return;
            }

            Vector<float, 3> sp = choose_split_point(node_idx);

            std::array<AABB<float>, 8> child_aabbs;
            const auto min_p = nodes[node_idx].aabb.min;
            const auto max_p = nodes[node_idx].aabb.max;
            for (int i = 0; i < 8; ++i) {
                child_aabbs[i].min = {
                    (i & 1) ? sp[0] : min_p[0], (i & 2) ? sp[1] : min_p[1], (i & 4) ? sp[2] : min_p[2]
                };
                child_aabbs[i].max = {
                    (i & 1) ? max_p[0] : sp[0], (i & 2) ? max_p[1] : sp[1], (i & 4) ? max_p[2] : sp[2]
                };
            }

            size_t start = nodes[node_idx].first_element;
            size_t end = start + nodes[node_idx].size;

            // --- NEW: Classify and separate keepers from movers ---
            std::vector<int> element_child_map(nodes[node_idx].size, -1); // Map local index to child index

            // First pass: find a home for each element
            for (size_t i = 0; i < nodes[node_idx].size; ++i) {
                size_t elem_idx = element_indices[start + i];
                const auto &elem_aabb = element_aabbs[elem_idx];

                for (int j = 0; j < 8; ++j) {
                    if (ContainsTraits<AABB<float>, AABB<float> >::contains(child_aabbs[j], elem_aabb)) {
                        element_child_map[i] = j; // Found a home
                        break; // Move to the next element
                    }
                }
                // If element_child_map[i] is still -1, it's a straddler
            }

            // In-place partition: move all elements that have a child home (movers) to the end of the range.
            // The `std::partition` will return an iterator to the first element of the "movers".
            auto movers_begin_it = std::partition(element_indices.begin() + start,
                                                  element_indices.begin() + end,
                                                  [&, i = 0](size_t) mutable {
                                                      return element_child_map[i++] == -1;
                                                      // Predicate: is this a keeper?
                                                  });

            // Now, update the parent node to only contain the "keepers".
            size_t num_keepers = std::distance(element_indices.begin() + start, movers_begin_it);
            nodes[node_idx].size = num_keepers;
            // parent_node.first_element remains the same.

            // The "movers" are in the range [movers_begin_it, element_indices.begin() + end)
            size_t movers_start_offset = std::distance(element_indices.begin(), movers_begin_it);
            size_t num_movers = std::distance(movers_begin_it, element_indices.begin() + end);

            // If we couldn't move any elements down, we must stop to prevent infinite recursion.
            if (num_movers == 0) {
                nodes[node_idx].is_leaf = true;
                return;
            }

            // --- NEW: Partition the movers ---
            // We need to re-scan the movers to find their child index.
            auto get_mover_child_idx = [&](size_t elem_idx) {
                const auto &elem_aabb = element_aabbs[elem_idx];
                for (int j = 0; j < 8; ++j) {
                    if (ContainsTraits<AABB<float>, AABB<float> >::contains(child_aabbs[j], elem_aabb)) {
                        return j;
                    }
                }
                return -1; // Should not happen for a mover
            };

            // 1. Count movers per child
            std::array<size_t, 8> child_sizes{};
            for (auto it = movers_begin_it; it != element_indices.begin() + end; ++it) {
                child_sizes[get_mover_child_idx(*it)]++;
            }

            // 2. Determine offsets for movers
            std::array<size_t, 8> child_offsets{};
            child_offsets[0] = movers_start_offset;
            for (int i = 1; i < 8; ++i) {
                child_offsets[i] = child_offsets[i - 1] + child_sizes[i - 1];
            }

            // 3. Re-arrange movers in-place using your scratch vector
            scratch_indices.assign(movers_begin_it, element_indices.begin() + end);

            std::array<size_t, 8> cur = child_offsets;
            for (size_t elem_idx: scratch_indices) {
                element_indices[cur[get_mover_child_idx(elem_idx)]++] = elem_idx;
            }

            // --- Create children and recurse (this logic is mostly the same) ---
            nodes[node_idx].is_leaf = false;
            for (int i = 0; i < 8; ++i) {
                if (child_sizes[i] > 0) {
                    const size_t child_node_idx = create_node();
                    nodes[node_idx].children[i] = child_node_idx;

                    Node &child = nodes[child_node_idx];
                    child.first_element = child_offsets[i];
                    child.size = child_sizes[i];


                    // If using tight_children policy, you could tighten it further here.
                    if (split_policy.tight_children) {
                        child.aabb = tight_child_aabb(element_indices.begin() + child.first_element,
                                                      element_indices.begin() + child.first_element + child.size,
                                                      split_policy.epsilon);
                    } else {
                        // Use the pre-calculated child AABB
                        child.aabb = child_aabbs[i];
                    }

                    subdivide_volume(child_node_idx, depth + 1);
                }
            }
        }

        [[nodiscard]] Vector<float, 3> compute_mean_center(size_t first, size_t size, const Vector<float, 3> &fallback_center) const {
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

        [[nodiscard]] Vector<float, 3> compute_median_center(size_t first, size_t size,
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

        [[nodiscard]] Vector<float, 3> choose_split_point(size_t node_idx) const {
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
        [[nodiscard]] AABB<float> tight_child_aabb(FwdIt begin, FwdIt end, float eps = 0.0f) const {
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

        [[nodiscard]] AABB<float> tight_child_aabb(const std::vector<size_t> &elems, float eps = 0.0f) const {
            return tight_child_aabb(elems.begin(), elems.end(), eps);
        }

        size_t max_elements_per_node = 10;
        size_t max_octree_depth = 10;
        SplitPolicy split_policy;
        std::vector<size_t> element_indices;
        std::vector<size_t> scratch_indices;
    };
}
