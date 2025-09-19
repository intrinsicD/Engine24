#pragma once

#include "Properties.h"
#include "AABB.h"
#include "Sphere.h"
#include "Logger.h"

#include <numeric>

namespace Bcg {
    class Octree {
    public:
        struct Node {
            size_t first_element = std::numeric_limits<size_t>::max();
            size_t last_element = std::numeric_limits<size_t>::max();
            size_t size = std::numeric_limits<size_t>::max();
            std::array<size_t, 8> children;
            bool is_leaf = true;

            Node() {
                children.fill(std::numeric_limits<size_t>::max());
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
        Property<AABB<float> > aabbs;
        size_t max_elements_per_node = 10;
        size_t max_depth = 10;

        Property<AABB<float> > element_aabbs;
        std::vector<size_t> element_indices;

        SplitPolicy split_policy;

        void build() {
            if (!element_aabbs) {
                Log::Error("Element AABBs property is not set. Cannot build octree.");
                return;
            }

            octree.clear();
            nodes = octree.add<Node>("n:nodes");
            aabbs = octree.add<AABB<float> >("n:aabbs");

            size_t num_elements = element_aabbs.vector().size();

            element_indices.resize(num_elements);
            std::iota(element_indices.begin(), element_indices.end(), 0);

            // Create root node
            size_t root_idx = create_node();
            nodes[root_idx].first_element = 0;
            nodes[root_idx].last_element = num_elements - 1;
            nodes[root_idx].size = num_elements;
            aabbs[root_idx] = AABB<float>::Build(element_aabbs.vector().begin(), element_aabbs.vector().end());

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

        void query_knn(const Eigen::Vector3f &query_point, size_t k, std::vector<size_t> &result) const {
            result.clear();
            if (octree.size() > 0) {
                query_recursive(0, query_point, k, result);
            }
        }

        void query_nearest(const Eigen::Vector3f &query_point, size_t &result) const {
            result = std::numeric_limits<size_t>::max();
            if (octree.size() > 0) {
                query_recursive(0, query_point, result);
            }
        }

    private:
        size_t create_node() {
            octree.push_back();
            return octree.size() - 1;
        }

        void subdivide(size_t node_idx, size_t depth) {
            Node &node = nodes[node_idx];

            if (depth >= max_depth || node.size <= max_elements_per_node) {
                node.is_leaf = true;
                return;
            }

            AABB<float> node_aabb = aabbs[node_idx];
            auto center = node_aabb.center();

            Eigen::Vector3f sp = choose_split_point(node_idx);

            // Create 8 child AABBs
            std::array<AABB<float>, 8> child_aabbs;
            auto min = node_aabb.min;
            auto max = node_aabb.max;

            for (int i = 0; i < 8; ++i) {
                child_aabbs[i].min = {
                    (i & 1) ? sp[0] : min[0],
                    (i & 2) ? sp[1] : min[1],
                    (i & 4) ? sp[2] : min[2]
                };
                child_aabbs[i].max = {
                    (i & 1) ? max[0] : sp[0],
                    (i & 2) ? max[1] : sp[1],
                    (i & 4) ? max[2] : sp[2]
                };
            }

            // Partition elements into children
            std::array<std::vector<size_t>, 8> child_elements;

            for (size_t i = node.first_element; i <= node.last_element; ++i) {
                size_t elem_idx = element_indices[i];
                AABB<float> elem_aabb = element_aabbs[elem_idx];
                auto elem_center = elem_aabb.center();

                int child_idx = 0;
                if (elem_center[0] >= sp[0]) child_idx |= 1;
                if (elem_center[1] >= sp[1]) child_idx |= 2;
                if (elem_center[2] >= sp[2]) child_idx |= 4;

                child_elements[child_idx].push_back(elem_idx);
            }

            // Create child nodes
            size_t current_pos = node.first_element;
            node.is_leaf = false;

            for (int i = 0; i < 8; ++i) {
                if (!child_elements[i].empty()) {
                    size_t child_idx = create_node();
                    node.children[i] = child_idx;

                    Node &child = nodes[child_idx];
                    child.first_element = current_pos;
                    child.size = child_elements[i].size();
                    child.last_element = current_pos + child.size - 1;

                    if (split_policy.tight_children) {
                        aabbs[child_idx] = tight_child_aabb(child_elements[i], split_policy.epsilon);
                    } else {
                        aabbs[child_idx] = child_aabbs[i];
                    }

                    // Copy elements to correct position
                    for (size_t j = 0; j < child.size; ++j) {
                        element_indices[current_pos + j] = child_elements[i][j];
                    }

                    current_pos += child.size;
                    subdivide(child_idx, depth + 1);
                }
            }
        }

        void query_recursive(size_t node_idx, const AABB<float> &query_aabb,
                             std::vector<size_t> &result) const {
            const Node &node = nodes[node_idx];
            const AABB<float>& nb = aabbs[node_idx];

            if (!IntersectsTraits<AABB<float>, AABB<float>>::intersects(nb, query_aabb)) {
                return;
            }

            const bool accept_all_safe = (split_policy.tight_children);
            if (accept_all_safe && ContainsTraits<AABB<float>, AABB<float>>::contains(query_aabb, nb)) {
                // Query fully contains the node: all elements intersect the query AABB.
                for (size_t i = node.first_element; i <= node.last_element; ++i) {
                    result.push_back(element_indices[i]);
                }
                return;
            }

            if (node.is_leaf) {
                for (size_t i = node.first_element; i <= node.last_element; ++i) {
                    size_t elem_idx = element_indices[i];
                    if (IntersectsTraits<AABB<float>, AABB<float>>::intersects(element_aabbs[elem_idx], query_aabb)) {
                        result.push_back(elem_idx);
                    }
                }
            } else {
                for (size_t child_idx : node.children) {
                    if (child_idx != std::numeric_limits<size_t>::max()) {
                        query_recursive(child_idx, query_aabb, result);
                    }
                }
            }
        }

        void query_recursive(size_t node_idx, const Sphere<float> &query_sphere,
                       std::vector<size_t> &result) const {
            const Node &node = nodes[node_idx];
            const AABB<float>& nb = aabbs[node_idx];

            // get center and radius (use radius() if your Sphere provides a getter)
            const auto qc = query_sphere.center();
            const float r  = query_sphere.radius;     // <-- change to query_sphere.radius() if needed
            const float r2 = r * r;

            // prune if no overlap using Minkowski sum logic
            if (!OverlapsTraits<Sphere<float>, AABB<float>>::overlaps(qc, r, r2, nb)) {
                return;
            }

            const bool accept_all_safe = (split_policy.tight_children || split_policy.assign_by_overlap);
            if (accept_all_safe && ContainsTraits<AABB<float>, Sphere<float>>::contains(qc, r2, nb)) {
                // Sphere fully contains the node's AABB => every element in this node intersects the sphere.
                for (size_t i = node.first_element; i <= node.last_element; ++i) {
                    result.push_back(element_indices[i]);
                }
                return; // early pruning
            }

            if (node.is_leaf) {
                for (size_t i = node.first_element; i <= node.last_element; ++i) {
                    const size_t elem_idx = element_indices[i];
                    if (IntersectsTraits<AABB<float>, Sphere<float>>::intersects(element_aabbs[elem_idx], query_sphere)) {
                        result.push_back(elem_idx);
                    }
                }
            } else {
                for (size_t child_idx : node.children) {
                    if (child_idx != std::numeric_limits<size_t>::max()) {
                        query_recursive(child_idx, query_sphere, result);
                    }
                }
            }
        }

        void query_recursive(size_t node_idx, const Eigen::Vector<float, 3> &query_point, size_t k,
                     std::vector<size_t> &result) const {

        }

        void query_recursive(size_t node_idx, const Eigen::Vector<float, 3> &query_point,
             std::vector<size_t> &result) const {

        }

        Eigen::Vector3f compute_mean_center(size_t first, size_t last) const {
            Eigen::Vector3f acc = Eigen::Vector3f::Zero();
            size_t n = last - first + 1;
            for (size_t i = first; i <= last; ++i) {
                const auto idx = element_indices[i];
                const auto c = element_aabbs[idx].center();
                acc += Eigen::Vector3f(c[0], c[1], c[2]);
            }
            return acc / float(n);
        }

        Eigen::Vector3f compute_median_center(size_t first, size_t last) const {
            const size_t n = last - first + 1;
            std::vector<float> xs;
            xs.reserve(n);
            std::vector<float> ys;
            ys.reserve(n);
            std::vector<float> zs;
            zs.reserve(n);
            for (size_t i = first; i <= last; ++i) {
                const auto idx = element_indices[i];
                const auto c = element_aabbs[idx].center();
                xs.push_back(c[0]);
                ys.push_back(c[1]);
                zs.push_back(c[2]);
            }
            auto kth = [](std::vector<float> &v) {
                const size_t k = v.size() / 2;
                std::nth_element(v.begin(), v.begin() + k, v.end());
                return v[k];
            };
            return {kth(xs), kth(ys), kth(zs)};
        }

        Eigen::Vector3f choose_split_point(size_t node_idx) const {
            const auto &node = nodes[node_idx];
            switch (split_policy.split_point) {
                case SplitPoint::Center: {
                    auto c = aabbs[node_idx].center();
                    return {c[0], c[1], c[2]};
                }
                case SplitPoint::Mean:
                    return compute_mean_center(node.first_element, node.last_element);
                case SplitPoint::Median:
                    return compute_median_center(node.first_element, node.last_element);
            }
            auto c = aabbs[node_idx].center();
            return {c[0], c[1], c[2]};
        }

        AABB<float> tight_child_aabb(const std::vector<size_t> &elems, float eps = 0.0f) const {
            if (elems.empty()) return {};
            AABB<float> tight = element_aabbs[elems[0]];
            for (size_t k = 1; k < elems.size(); ++k) {
                tight.merge(element_aabbs[elems[k]]);
            }
            if (eps != 0.0f) {
                tight.min[0] -= eps;
                tight.min[1] -= eps;
                tight.min[2] -= eps;
                tight.max[0] += eps;
                tight.max[1] += eps;
                tight.max[2] += eps;
            }
            return tight;
        }
    };
}
