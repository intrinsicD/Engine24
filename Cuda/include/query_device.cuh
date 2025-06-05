//
// Created by alex on 30.05.25.
//

#ifndef ENGINE24_QUERY_DEVICE_CUH
#define ENGINE24_QUERY_DEVICE_CUH

#include "bvh_device.cuh"
#include "predicator.cuh"
#include <queue>

namespace Bcg::cuda::bvh {
    /* Query the BVH for objects that overlap with the given query object, e.g., an AABB or a sphere.
     * The results are written to the output iterator.
     * Returns the number of objects found.
     */

    template<typename ObjectType, typename QueryObjectType>
    __device__
    inline unsigned int query_device(const detail::basic_device_bvh<ObjectType, true> &bvh,
                                     const query_overlap_count <QueryObjectType> &query) noexcept {
        unsigned int stack[64];
        unsigned int *stack_ptr = stack;
        *stack_ptr++ = 0; // Start with the root node

        unsigned int num_found = 0;
        do {
            const unsigned int node_index = *--stack_ptr;
            const detail::node &node = bvh.nodes[node_index];

            const unsigned int L_idx = node.left_idx;
            const unsigned int R_idx = node.right_idx;

            bool intersects_left = intersects(query.target, bvh.aabbs[L_idx]);
            bool intersects_right = intersects(query.target, bvh.aabbs[R_idx]);

            if (intersects_left) {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) {
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
        } while (stack < stack_ptr);

        return num_found;
    }

    template<typename ObjectType, typename QueryObjectType, typename OutputIterator>
    __device__
    inline unsigned int query_device(const detail::basic_device_bvh<ObjectType, true> &bvh,
                                     const query_overlap <QueryObjectType> &query,
                                     OutputIterator outiter,
                                     const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept {
        unsigned int stack[64];
        unsigned int *stack_ptr = stack;
        *stack_ptr++ = 0; // Start with the root node

        unsigned int num_found = 0;
        do {
            const unsigned int node_index = *--stack_ptr;
            const detail::node &node = bvh.nodes[node_index];

            const unsigned int L_idx = node.left_idx;
            const unsigned int R_idx = node.right_idx;

            bool intersects_left = intersects(query.target, bvh.aabbs[L_idx]);
            bool intersects_right = intersects(query.target, bvh.aabbs[R_idx]);

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

    /* Query the BVH for the closest object to the given query point.
     * Returns the pair<index, distance> of the closest object found.
     */

    template<typename ObjectType, typename DistanceCalculator>
    __device__
    inline thrust::pair<unsigned int, float> query_device(const detail::basic_device_bvh<ObjectType, true> &bvh,
                                                          const query_nearest &query,
                                                          DistanceCalculator calc_dist) noexcept {
        using index_type = unsigned int;
        thrust::pair<unsigned int, float> stack[64];
        thrust::pair<unsigned int, float> *stack_ptr = stack;
        *stack_ptr++ = thrust::make_pair(0, mindist(bvh.aabbs[0], query.target));

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

            const float L_mindist = mindist(L_box, query.target);
            const float R_mindist = mindist(R_box, query.target);

            const float L_minmaxdist = minmaxdist(L_box, query.target);
            const float R_minmaxdist = minmaxdist(R_box, query.target);

            // there should be an object that locates within minmaxdist.

            if (L_mindist <= R_minmaxdist) // L is worth considering
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    const float dist = calc_dist(query.target, bvh.objects[obj_idx]);
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
                    const float dist = calc_dist(query.target, bvh.objects[obj_idx]);
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

    /* Query the BVH for the k closest objects to the given query point.
     * The results are written to the output iterator.
     * Returns the number of objects found.
     */

    template<typename ObjectType, typename DistanceCalculator>
    __device__
    inline unsigned int query_device(const detail::basic_device_bvh<ObjectType, true> &bvh,
                                     const query_knn &query,
                                     DistanceCalculator calc_dist,
                                     thrust::pair<unsigned int, float> *results) {
        using index_type = unsigned int;
        // Array to store the k-nearest neighbors and their distances

        const unsigned int k = query.k_closest;

        // Initialize results with invalid indices and maximum distances
        for (unsigned int i = 0; i < k; ++i) {
            *(results + i) = thrust::make_pair<unsigned int, float>(0xFFFFFFFF, INFINITY);  // Invalid index
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
            float dist_L = mindist(bvh.aabbs[L_idx], query.target);
            float dist_R = mindist(bvh.aabbs[R_idx], query.target);

            // Push children to stack if their distance is less than the farthest current distance
            bool push_L = (L_idx != 0xFFFFFFFF) && (dist_L < results[0].second);
            bool push_R = (R_idx != 0xFFFFFFFF) && (dist_R < results[0].second);

            // Handle left child
            if (push_L) {
                if (bvh.nodes[L_idx].object_idx != 0xFFFFFFFF) {
                    // Leaf node: calculate distance to object
                    float obj_dist = calc_dist(query.target, bvh.objects[bvh.nodes[L_idx].object_idx]);

                    if (obj_dist < results[0].second) {
                        // Insert into result set and maintain max-heap property
                        results[0].second = obj_dist;
                        results[0].first = bvh.nodes[L_idx].object_idx;

                        // Bubble down to maintain max-heap
                        for (int i = 1; i < k; ++i) {
                            if (results[i - 1].second < results[i].second) {
                                thrust::swap(results[i - 1], results[i]);
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
                    float obj_dist = calc_dist(query.target, bvh.objects[bvh.nodes[R_idx].object_idx]);

                    if (obj_dist < results[0].second) {
                        // Insert into result set and maintain max-heap property
                        results[0].second = obj_dist;
                        results[0].first = bvh.nodes[R_idx].object_idx;

                        // Bubble down to maintain max-heap
                        for (int i = 1; i < k; ++i) {
                            if (results[i - 1].second < results[i].second) {
                                thrust::swap(results[i - 1], results[i]);
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
        for (unsigned int i = 0; i < k; ++i) {
            if ((results + i)->first != 0xFFFFFFFF) {
                ++num_found;
            }
        }

        return num_found;
    }

    template<typename ObjectType, typename QueryObjectType>
    __global__
    void query_overlap_count_parallel_kernel( // Renamed for clarity
            const detail::basic_device_bvh<ObjectType, true> &bvh,
            const QueryObjectType *d_query_targets,
            unsigned int num_total_queries, // Total number of queries
            unsigned int *d_num_found_output_buffer // Master output buffer
    ) {
        unsigned int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (query_idx < num_total_queries) {
            auto current_query_object = d_query_targets[query_idx];
            query_overlap_count<QueryObjectType> current_query(current_query_object);

            unsigned int num_found = query_device(bvh, current_query);
            d_num_found_output_buffer[query_idx] = num_found;
        }
    }

    template<typename ObjectType, typename QueryObjectType>
    __global__
    void query_overlap_parallel_kernel( // Renamed for clarity
            const detail::basic_device_bvh<ObjectType, true> &bvh,
            const QueryObjectType *d_query_targets,
            const unsigned int *d_result_offsets, // Optional: to store offsets for each query
            const unsigned int *d_num_found_per_query, // Optional: to store actual counts
            unsigned int num_total_queries, // Total number of queries
            unsigned int *d_all_results_output_buffer // Master output buffer
    ) {
        unsigned int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (query_idx < num_total_queries) {
            auto current_query_object = d_query_targets[query_idx];
            query_overlap<QueryObjectType> current_query(current_query_object);

            unsigned int max_buffer_size = d_num_found_per_query ? d_num_found_per_query[query_idx] : 0xFFFFFFFF;

            auto num_found = query_device(bvh, current_query,
                                          d_all_results_output_buffer + d_result_offsets[query_idx],
                                          max_buffer_size);

            // check if num_found is equal to the max buffer size
            if (max_buffer_size != num_found) {
                printf("num_found (%u) != max_buffer_size (%u) for query %u\n", num_found, max_buffer_size, query_idx);
            }
        }
    }


    template<typename ObjectType, typename DistanceCalculator>
    __global__
    void query_nearest_parallel_kernel( // Renamed for clarity
            const detail::basic_device_bvh<ObjectType, true> &bvh,
            const ObjectType *d_query_targets,      // Array of query target points/objects
            const unsigned int num_total_queries,
            DistanceCalculator calc_dist,
            thrust::pair<unsigned int, float> *d_all_results_output_buffer // Master output buffer
    ) {
        unsigned int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (query_idx < num_total_queries) {

            // Construct the query_knn object for the current target
            query_nearest current_query_object(d_query_targets[query_idx]);

            thrust::pair<unsigned int, float> result = query_device( // Call the device function
                    bvh,
                    current_query_object,
                    calc_dist
            );

            if (d_all_results_output_buffer != nullptr) {
                d_all_results_output_buffer[query_idx] = result;
            }
        }
    }

    template<typename ObjectType, typename DistanceCalculator>
    __global__
    void query_knn_parallel_fixed_k_kernel( // Renamed for clarity
            const detail::basic_device_bvh<ObjectType, true> &bvh,
            const ObjectType *d_query_targets,      // Array of query target points/objects
            const unsigned int num_total_queries,
            const unsigned int k_fixed_for_all_queries, // The K for KNN, same for all
            DistanceCalculator calc_dist,
            thrust::pair<unsigned int, float> *d_all_results_output_buffer, // Master output buffer
            unsigned int *d_num_found_per_query      // Optional: to store actual counts
            // const unsigned int max_buffer_size is implicitly checked by caller during allocation
    ) {
        unsigned int query_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (query_idx < num_total_queries) {
            if (k_fixed_for_all_queries == 0) {
                if (d_num_found_per_query != nullptr) {
                    d_num_found_per_query[query_idx] = 0;
                }
                return;
            }

            // Construct the query_knn object for the current target
            query_knn current_query_object(d_query_targets[query_idx], k_fixed_for_all_queries);

            // Calculate the output pointer for this query's results
            thrust::pair<unsigned int, float> *thread_results_slice_ptr =
                    d_all_results_output_buffer + (query_idx * k_fixed_for_all_queries);

            unsigned int num_found = query_device( // Call the device function
                    bvh,
                    current_query_object,
                    calc_dist,
                    thread_results_slice_ptr
            );

            if (d_num_found_per_query != nullptr) {
                d_num_found_per_query[query_idx] = num_found;
            }
        }
    }

    template<typename ObjectType, typename QueryObjectType>
    __host__
    inline void query_host(const host_data<ObjectType> &bvh,
                           const query_overlap <QueryObjectType> &query,
                           std::vector<unsigned int> &outiter) noexcept {

        using index_type = unsigned int;
        std::vector<std::size_t> stack;
        stack.reserve(128);
        stack.push_back(0);

        outiter.clear();
        do {
            const index_type node = stack.back();
            stack.pop_back();

            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            bool intersects_left = intersects(query.target, bvh.aabbs[L_idx]);
            bool intersects_right = intersects(query.target, bvh.aabbs[R_idx]);

            if (intersects_left) {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) {
                    outiter.push_back(obj_idx);
                } else // the node is not a leaf.
                {
                    stack.push_back(L_idx);
                }
            }

            if (intersects_right) {
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) {
                    outiter.push_back(obj_idx);
                } else // the node is not a leaf.
                {
                    stack.push_back(R_idx);
                }
            }
        } while (!stack.empty());
    }

    template<typename ObjectType, typename DistanceCalculator>
    __host__
    inline std::pair<unsigned int, float> query_host(const host_data<ObjectType> &bvh,
                                                     const query_nearest &query,
                                                     DistanceCalculator calc_dist) noexcept {
        using index_type = unsigned int;
        // pair of {node_idx, mindist}
        std::vector<std::pair<index_type, float>> stack = {
                {0, mindist(bvh.aabbs[0], query.target)}
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

            const index_type L_idx = bvh.nodes[node.first].left_idx;
            const index_type R_idx = bvh.nodes[node.first].right_idx;

            const aabb &L_box = bvh.aabbs[L_idx];
            const aabb &R_box = bvh.aabbs[R_idx];

            const float L_mindist = mindist(L_box, query.target);
            const float R_mindist = mindist(R_box, query.target);

            const float L_minmaxdist = minmaxdist(L_box, query.target);
            const float R_minmaxdist = minmaxdist(R_box, query.target);

            // there should be an object that locates within minmaxdist.

            if (L_mindist <= R_minmaxdist) // L is worth considering
            {
                const auto obj_idx = bvh.nodes[L_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    const float dist = calc_dist(query.target, bvh.objects[obj_idx]);
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
                const auto obj_idx = bvh.nodes[R_idx].object_idx;
                if (obj_idx != 0xFFFFFFFF) // leaf node
                {
                    const float dist = calc_dist(query.target, bvh.objects[obj_idx]);
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

    template<typename ObjectType, typename DistanceCalculator>
    __host__
    inline void query_host(const host_data<ObjectType> &bvh,
                           const query_knn &query,
                           DistanceCalculator calc_dist,
                           std::vector<std::pair<unsigned int, float>> &results) {
        using index_type = unsigned int;
        // Array to store the k-nearest neighbors and their distances

        const unsigned int k = query.k_closest;

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

            const index_type L_idx = bvh.nodes[node].left_idx;
            const index_type R_idx = bvh.nodes[node].right_idx;

            const auto &L_aabb = bvh.aabbs[L_idx];
            const auto &R_aabb = bvh.aabbs[R_idx];

            // Check left child
            if (L_idx != 0xFFFFFFFF) {
                float dist = mindist(L_aabb, query.target);
                if (knn_heap.size() < k || dist < knn_heap.top().first) {
                    if (bvh.nodes[L_idx].object_idx != 0xFFFFFFFF) {
                        // Leaf node, calculate the distance to the object
                        index_type obj_idx = bvh.nodes[L_idx].object_idx;
                        float obj_dist = calc_dist(query.target, bvh.objects[obj_idx]);
                        if (knn_heap.size() < k) {
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
                float dist = mindist(R_aabb, query.target);
                if (knn_heap.size() < k || dist < knn_heap.top().first) {
                    if (bvh.nodes[R_idx].object_idx != 0xFFFFFFFF) {
                        // Leaf node, calculate the distance to the object
                        index_type obj_idx = bvh.nodes[R_idx].object_idx;
                        float obj_dist = calc_dist(query.target, bvh.objects[obj_idx]);
                        if (knn_heap.size() < k) {
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
        results.resize(k);
        for (unsigned int i = 0; i < k; ++i) {
            results[i] = knn_heap.top();
            knn_heap.pop();
            if (knn_heap.empty()) break;
        }
    }

}

#endif //ENGINE24_QUERY_DEVICE_CUH
