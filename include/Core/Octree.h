#pragma once


#include "Properties.h"
#include "AABB.h"
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

        PropertyContainer octree;
        Property<Node> nodes;
        Property<AABB<float> > aabbs;
        size_t max_elements_per_node = 10;
        size_t max_depth = 10;

        Property<AABB<float> > element_aabbs;
        std::vector<size_t> element_indices;

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

            // Create 8 child AABBs
            std::array<AABB<float>, 8> child_aabbs;
            auto min = node_aabb.min;
            auto max = node_aabb.max;

            for (int i = 0; i < 8; ++i) {
                child_aabbs[i].min = {
                    (i & 1) ? center[0] : min[0],
                    (i & 2) ? center[1] : min[1],
                    (i & 4) ? center[2] : min[2]
                };
                child_aabbs[i].max = {
                    (i & 1) ? max[0] : center[0],
                    (i & 2) ? max[1] : center[1],
                    (i & 4) ? max[2] : center[2]
                };
            }

            // Partition elements into children
            std::array<std::vector<size_t>, 8> child_elements;

            for (size_t i = node.first_element; i <= node.last_element; ++i) {
                size_t elem_idx = element_indices[i];
                AABB<float> elem_aabb = element_aabbs[elem_idx];
                auto elem_center = elem_aabb.center();

                int child_idx = 0;
                if (elem_center[0] >= center[0]) child_idx |= 1;
                if (elem_center[1] >= center[1]) child_idx |= 2;
                if (elem_center[2] >= center[2]) child_idx |= 4;

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
                    aabbs[child_idx] = child_aabbs[i];

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

            if (!IntersectsTraits<AABB<float>, AABB<float> >::intersects(aabbs[node_idx], query_aabb)) {
                return;
            }

            if (node.is_leaf) {
                for (size_t i = node.first_element; i <= node.last_element; ++i) {
                    size_t elem_idx = element_indices[i];
                    if (IntersectsTraits<AABB<float>, AABB<float> >::intersects(element_aabbs[elem_idx], query_aabb)) {
                        result.push_back(elem_idx);
                    }
                }
            } else {
                for (size_t child_idx: node.children) {
                    if (child_idx != std::numeric_limits<size_t>::max()) {
                        query_recursive(child_idx, query_aabb, result);
                    }
                }
            }
        }
    };
}
