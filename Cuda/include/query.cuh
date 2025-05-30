#ifndef LBVH_QUERY_CUH
#define LBVH_QUERY_CUH

#include "predicator.cuh"
#include "utility.cuh"
#include "heap.cuh"
#include "bvh.cuh"
#include <thrust/sort.h>
#include <queue>

namespace Bcg::cuda {
    // query object indices that potentially overlaps with query aabb.
    //
    // requirements:
    // - OutputIterator should be writable and its object_type should be uint32_t
    //
    template<typename Objects, bool IsConst, typename QueryObject, typename OutputIterator>
    __device__
    unsigned int query_device(
            const detail::basic_device_bvh<Objects, IsConst> &bvh,
            const query_overlap<QueryObject> q, OutputIterator outiter,
            const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept {
        using bvh_type = detail::basic_device_bvh<Objects, IsConst>;
        using index_type = typename bvh_type::index_type;

        index_type stack[64]; // Consider stack size in relation to BVH depth.
        index_type *stack_ptr = stack;
        *stack_ptr++ = 0; // root node is always 0

        unsigned int num_found = 0;
        do {
            const index_type node = *--stack_ptr;
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            bool intersects_left = intersects(q.target, bvh.aabbs[L_idx]);
            bool intersects_right = intersects(q.target, bvh.aabbs[R_idx]);

            if (intersects_left) {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) {
                    if (num_found < max_buffer_size) {
                        *outiter++ = obj_idx;
                    }
                    ++num_found;
                } else {
                    if (stack_ptr - stack >= 64) {
                        // Handle stack overflow, e.g., return early or log an error.
                        printf("stack overflow\n");
                        return num_found;
                    }
                    *stack_ptr++ = L_idx;
                }
            }

            if (intersects_right) {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) {
                    if (num_found < max_buffer_size) {
                        *outiter++ = obj_idx;
                    }
                    ++num_found;
                } else {
                    if (stack_ptr - stack >= 64) {
                        // Handle stack overflow, e.g., return early or log an error.
                        printf("stack overflow\n");
                        return num_found;
                    }
                    *stack_ptr++ = R_idx;
                }
            }

            if (num_found >= max_buffer_size) {
                printf("num_found >= max_buffer_size\n");
                break;
            }
        } while (stack < stack_ptr);

        return num_found;
    }

    // query object index that is the nearst to the query point.
    //
    // requirements:
    // - DistanceCalculator must be able to calc distance between a point to an object.
    //
    template<typename Objects, bool IsConst, typename DistanceCalculator>
    __device__
    thrust::pair<unsigned int, float> query_device(
            const detail::basic_device_bvh<Objects, IsConst> &bvh,
            const query_nearest &q, DistanceCalculator calc_dist) noexcept {
        using bvh_type = detail::basic_device_bvh<Objects, IsConst>;
        using index_type = typename bvh_type::index_type;

        // pair of {node_idx, mindist}
        thrust::pair<index_type, float> stack[64];
        thrust::pair<index_type, float> *stack_ptr = stack;
        *stack_ptr++ = thrust::make_pair(0, mindist(bvh.aabbs[0], q.target));

        unsigned int nearest = 0xFFFFFFFF;
        float dist_to_nearest_object = infinity<float>();
        do {
            const auto node = *--stack_ptr;
            if (node.second > dist_to_nearest_object) {
                // if aabb mindist > already_found_mindist, it cannot have a nearest
                continue;
            }

            const index_type L_idx = bvh.nodes[node.first].left_idx;
            const index_type R_idx = bvh.nodes[node.first].right_idx;

            const aabb &L_box = bvh.aabbs[L_idx];
            const aabb &R_box = bvh.aabbs[R_idx];

            const float L_mindist = mindist(L_box, q.target);
            const float R_mindist = mindist(R_box, q.target);

            const float L_minmaxdist = minmaxdist(L_box, q.target);
            const float R_minmaxdist = minmaxdist(R_box, q.target);

            // there should be an object that locates within minmaxdist.

            if (L_mindist <= R_minmaxdist) // L is worth considering
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    const float dist = calc_dist(q.target, bvh.objects[obj_idx]);
                    if (dist <= dist_to_nearest_object) {
                        dist_to_nearest_object = dist;
                        nearest = obj_idx;
                    }
                } else {
                    *stack_ptr++ = thrust::make_pair(L_idx, L_mindist);
                }
            }
            if (R_mindist <= L_minmaxdist) // R is worth considering
            {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    const float dist = calc_dist(q.target, bvh.objects[obj_idx]);
                    if (dist <= dist_to_nearest_object) {
                        dist_to_nearest_object = dist;
                        nearest = obj_idx;
                    }
                } else {
                    *stack_ptr++ = thrust::make_pair(R_idx, R_mindist);
                }
            }
            assert(stack_ptr < stack + 64);
        } while (stack < stack_ptr);
        return thrust::make_pair(nearest, dist_to_nearest_object);
    }


    // query object index that is the nearst to the query point.
    //
    // requirements:
    // - DistanceCalculator must be able to calc distance between a point to an object.
    //
    template<typename Objects, bool IsConst, typename DistanceCalculator, typename OutputIterator>
    __device__ unsigned int query_device(
            const detail::basic_device_bvh<Objects, IsConst> &bvh,
            const query_knn &q, DistanceCalculator calc_dist,
            OutputIterator outiter,
            const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept {

        using index_type = typename detail::basic_device_bvh<Objects, true>::index_type;

        // Array to store the k-nearest neighbors and their distances
        unsigned int result[32];
        float distances[32];

        const unsigned int k = q.k_closest;

        // Initialize results with invalid indices and maximum distances
        for (unsigned int i = 0; i < k; ++i) {
            result[i] = 0xFFFFFFFF;  // Invalid index
            distances[i] = INFINITY;  // Maximum possible distance
        }

        // Traversal stack
        int stack[64];
        int stack_ptr = 0;
        stack[stack_ptr++] = 0;  // Start with root node

        while (stack_ptr > 0) {
            const int node = stack[--stack_ptr];

            // Get child indices
            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            // Calculate minimum distances to child AABBs
            float dist_L = mindist(bvh.aabbs[L_idx], q.target);
            float dist_R = mindist(bvh.aabbs[R_idx], q.target);

            // Push children to stack if their distance is less than the farthest current distance
            bool push_L = (L_idx != 0xFFFFFFFF) && (dist_L < distances[0]);
            bool push_R = (R_idx != 0xFFFFFFFF) && (dist_R < distances[0]);

            // Handle left child
            if (push_L) {
                if (bvh.nodes[L_idx].object_idx != 0xFFFFFFFF) {
                    // Leaf node: calculate distance to object
                    float obj_dist = calc_dist(q.target, bvh.objects[bvh.nodes[L_idx].object_idx]);

                    if (obj_dist < distances[0]) {
                        // Insert into result set and maintain max-heap property
                        distances[0] = obj_dist;
                        result[0] = bvh.nodes[L_idx].object_idx;

                        // Bubble down to maintain max-heap
                        for (int i = 1; i < k; ++i) {
                            if (distances[i - 1] < distances[i]) {
                                thrust::swap(distances[i - 1], distances[i]);
                                thrust::swap(result[i - 1], result[i]);
                            }
                        }
                    }
                } else {
                    stack[stack_ptr++] = L_idx;
                }
            }

            // Handle right child
            if (push_R) {
                if (bvh.nodes[R_idx].object_idx != 0xFFFFFFFF) {
                    // Leaf node: calculate distance to object
                    float obj_dist = calc_dist(q.target, bvh.objects[bvh.nodes[R_idx].object_idx]);

                    if (obj_dist < distances[0]) {
                        // Insert into result set and maintain max-heap property
                        distances[0] = obj_dist;
                        result[0] = bvh.nodes[R_idx].object_idx;

                        // Bubble down to maintain max-heap
                        for (int i = 1; i < k; ++i) {
                            if (distances[i - 1] < distances[i]) {
                                thrust::swap(distances[i - 1], distances[i]);
                                thrust::swap(result[i - 1], result[i]);
                            }
                        }
                    }
                } else {
                    stack[stack_ptr++] = R_idx;
                }
            }
        }

        // Copy results to output iterator
        unsigned int num_found = 0;
        for (unsigned int i = 0; i < k && i < max_buffer_size; ++i) {
            if (result[i] != 0xFFFFFFFF) {
                *outiter++ = result[i];
                ++num_found;
            }
        }

        return num_found;
    }


    template<typename Objects, typename AABBGetter,
            typename MortonCodeCalculator, typename QueryObject>
    unsigned int query_host(
            const lbvh<Objects, AABBGetter, MortonCodeCalculator> &tree,
            const query_overlap<QueryObject> q, std::vector<size_t> &outiter) {
        using bvh_type = lbvh<Objects, AABBGetter, MortonCodeCalculator>;
        using index_type = typename bvh_type::index_type;

        if (!tree.query_host_enabled()) {
            throw std::runtime_error("lbvh::bvh query_host is not enabled");
        }

        std::vector<std::size_t> stack;
        stack.reserve(128);
        stack.push_back(0);

        unsigned int num_found = 0;
        do {
            const index_type node = stack.back();
            stack.pop_back();
            const index_type L_idx = tree.nodes_host()[node].left_idx;
            const index_type R_idx = tree.nodes_host()[node].right_idx;

            if (L_idx < tree.nodes_host().size() && intersects(q.target, tree.aabbs_host()[L_idx])) {
                const auto obj_idx = tree.nodes_host()[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) {
                    outiter.push_back(obj_idx);
                    ++num_found;
                } else // the node is not a leaf.
                {
                    stack.push_back(L_idx);
                }
            }
            if (R_idx < tree.nodes_host().size() && intersects(q.target, tree.aabbs_host()[R_idx])) {
                const auto obj_idx = tree.nodes_host()[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) {
                    outiter.push_back(obj_idx);
                    ++num_found;
                } else // the node is not a leaf.
                {
                    stack.push_back(R_idx);
                }
            }
        } while (!stack.empty());
        return num_found;
    }

    template<typename Objects, typename AABBGetter,
            typename MortonCodeCalculator, typename DistanceCalculator>
    std::pair<unsigned int, float> query_host(
            const lbvh<Objects, AABBGetter, MortonCodeCalculator> &tree,
            const query_nearest &q, DistanceCalculator calc_dist)  {
        using bvh_type = lbvh<Objects, AABBGetter, MortonCodeCalculator>;
        using index_type = typename bvh_type::index_type;

        if (!tree.query_host_enabled()) {
            throw std::runtime_error("lbvh::bvh query_host is not enabled");
        }

        // pair of {node_idx, mindist}
        std::vector<std::pair<index_type, float>> stack = {
                {0, mindist(tree.aabbs_host()[0], q.target)}
        };
        stack.reserve(64);

        unsigned int nearest = 0xFFFFFFFF;
        float current_nearest_dist = infinity<float>();
        do {
            const auto node = stack.back();
            stack.pop_back();
            if (node.second > current_nearest_dist) {
                // if aabb mindist > already_found_mindist, it cannot have a nearest
                continue;
            }

            const index_type L_idx = tree.nodes_host()[node.first].left_idx;
            const index_type R_idx = tree.nodes_host()[node.first].right_idx;

            const aabb &L_box = tree.aabbs_host()[L_idx];
            const aabb &R_box = tree.aabbs_host()[R_idx];

            const float L_mindist = mindist(L_box, q.target);
            const float R_mindist = mindist(R_box, q.target);

            const float L_minmaxdist = minmaxdist(L_box, q.target);
            const float R_minmaxdist = minmaxdist(R_box, q.target);

            // there should be an object that locates within minmaxdist.

            if (L_mindist <= R_minmaxdist) // L is worth considering
            {
                const auto obj_idx = tree.nodes_host()[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    const float dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                    if (dist <= current_nearest_dist) {
                        current_nearest_dist = dist;
                        nearest = obj_idx;
                    }
                } else {
                    stack.emplace_back(L_idx, L_mindist);
                }
            }
            if (R_mindist <= L_minmaxdist) // R is worth considering
            {
                const auto obj_idx = tree.nodes_host()[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    const float dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                    if (dist <= current_nearest_dist) {
                        current_nearest_dist = dist;
                        nearest = obj_idx;
                    }
                } else {
                    stack.emplace_back(R_idx, R_mindist);
                }
            }
        } while (!stack.empty());
        return std::make_pair(nearest, current_nearest_dist);
    }

    template<typename Objects, typename AABBGetter,
            typename MortonCodeCalculator, typename DistanceCalculator>
    std::vector<unsigned int> query_host(
            const lbvh<Objects, AABBGetter, MortonCodeCalculator> &tree,
            const query_knn &q, DistanceCalculator calc_dist,
            const unsigned int max_buffer_size = 0xFFFFFFFF) {
        using index_type = typename lbvh<Objects, AABBGetter, MortonCodeCalculator>::index_type;

        if (!tree.query_host_enabled()) {
            throw std::runtime_error("lbvh::bvh query_host is not enabled");
        }

        // Priority queue for KNN, max-heap
        auto knn_compare = [](const std::pair<float, index_type> &lhs,
                              const std::pair<float, index_type> &rhs) {
            return lhs.first < rhs.first; // max-heap, so closer (smaller) distances should be at the top
        };
        std::priority_queue<std::pair<float, index_type>,
                std::vector<std::pair<float, index_type>>,
                decltype(knn_compare)> knn_heap(knn_compare);

        // Priority queue for nodes, min-heap
        auto node_compare = [](const std::pair<float, index_type> &lhs,
                               const std::pair<float, index_type> &rhs) {
            return lhs.first > rhs.first; // min-heap, so the closest node (smaller distance) is explored first
        };
        std::priority_queue<std::pair<float, index_type>,
                std::vector<std::pair<float, index_type>>,
                decltype(node_compare)> node_heap(node_compare);

        // Start with the root node
        node_heap.emplace(0.0f, 0);

        while (!node_heap.empty()) {
            const auto [current_dist, node] = node_heap.top();
            node_heap.pop();

            const index_type L_idx = tree.nodes_host()[node].left_idx;
            const index_type R_idx = tree.nodes_host()[node].right_idx;

            const auto &L_aabb = tree.aabbs_host()[L_idx];
            const auto &R_aabb = tree.aabbs_host()[R_idx];

            // Check left child
            if (L_idx != 0xFFFFFFFF) {
                float dist = mindist(L_aabb, q.target);
                if (knn_heap.size() < q.k_closest || dist < knn_heap.top().first) {
                    if (tree.nodes_host()[L_idx].object_idx != 0xFFFFFFFF) {
                        // Leaf node, calculate the distance to the object
                        index_type obj_idx = tree.nodes_host()[L_idx].object_idx;
                        float obj_dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                        if (knn_heap.size() < q.k_closest) {
                            knn_heap.emplace(obj_dist, obj_idx);
                        } else if (obj_dist < knn_heap.top().first) {
                            knn_heap.pop();
                            knn_heap.emplace(obj_dist, obj_idx);
                        }
                    } else {
                        // Internal node, push it to the node heap
                        node_heap.emplace(dist, L_idx);
                    }
                }
            }

            // Check right child
            if (R_idx != 0xFFFFFFFF) {
                float dist = mindist(R_aabb, q.target);
                if (knn_heap.size() < q.k_closest || dist < knn_heap.top().first) {
                    if (tree.nodes_host()[R_idx].object_idx != 0xFFFFFFFF) {
                        // Leaf node, calculate the distance to the object
                        index_type obj_idx = tree.nodes_host()[R_idx].object_idx;
                        float obj_dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                        if (knn_heap.size() < q.k_closest) {
                            knn_heap.emplace(obj_dist, obj_idx);
                        } else if (obj_dist < knn_heap.top().first) {
                            knn_heap.pop();
                            knn_heap.emplace(obj_dist, obj_idx);
                        }
                    } else {
                        // Internal node, push it to the node heap
                        node_heap.emplace(dist, R_idx);
                    }
                }
            }
        }

        // Extract the results from the KNN heap
        std::vector<unsigned int> result;
        result.reserve(q.k_closest);
        while (!knn_heap.empty() && result.size() < max_buffer_size) {
            result.push_back(knn_heap.top().second);
            knn_heap.pop();
        }

        return result;
    }

} // lbvh
#endif// LBVH_QUERY_CUH
