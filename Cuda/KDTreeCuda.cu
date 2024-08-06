//
// Created by alex on 06.08.24.
//

#include <thrust/heap.h>
#include "KDTreeCuda.cuh"

namespace Bcg::Cuda {
    __global__ void Cuda::build_kdtree(KDNode *nodes, float *points, int depth, int start, int end) {
        if (start >= end) return;

        int axis = depth % 3;
        int mid = (start + end) / 2;

        thrust::sort(thrust::device_pointer_cast(points + start * 3), thrust::device_pointer_cast(points + end * 3),
        [axis] __device__(const float &a,
        const float &b) {
            return a[axis] < b[axis];
        });

        // Set node values
        nodes[mid].split_value = points[mid * 3 + axis];
        nodes[mid].index = mid;
        nodes[mid].left = (start < mid) ? mid - 1 : -1;
        nodes[mid].right = (mid < end - 1) ? mid + 1 : -1;

        // Recursively build left and right subtrees
        build_kdtree<<<1, 1>>>(nodes, points, depth + 1, start, mid);
        build_kdtree<<<1, 1>>>(nodes, points, depth + 1, mid + 1, end);
    }

    struct Neighbor {
        float distance;
        int index;

        __device__ __host__ bool operator<(const Neighbor &other) const {
            return distance < other.distance;
        }
    };

    __device__ float euclidean_distance(const float *a, const float *b, int dims) {
        float dist = 0.0f;
        for (int i = 0; i < dims; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return sqrt(dist);
    }

    __device__ void
    knn_search(const KDNode *nodes, const float *points, const float *query, int node_idx, int depth, int k,
               Neighbor *best_neighbors, int &best_count) {
        if (node_idx == -1) return;

        const KDNode &node = nodes[node_idx];
        int axis = depth % 3;

        // Compute distance to current point
        float distance = euclidean_distance(query, points + node.index * 3, 3);
        if (best_count < k || distance < best_neighbors[0].distance) {
            if (best_count == k) {
                thrust::pop_heap(best_neighbors, best_neighbors + k);
            } else {
                ++best_count;
            }
            best_neighbors[best_count - 1] = {distance, node.index};
            thrust::push_heap(best_neighbors, best_neighbors + best_count);
        }

        // Determine which side of the split plane to search
        float diff = query[axis] - points[node.index * 3 + axis];
        int near_idx = diff < 0 ? node.left : node.right;
        int far_idx = diff < 0 ? node.right : node.left;

        // Recursively search the nearer side of the split plane
        knn_search(nodes, points, query, near_idx, depth + 1, k, best_neighbors, best_count);

        // If there's potential for the farther side to contain closer neighbors, search it as well
        if (best_count < k || fabs(diff) < best_neighbors[0].distance) {
            knn_search(nodes, points, query, far_idx, depth + 1, k, best_neighbors, best_count);
        }
    }

    __global__ void Cuda::knn_query(KDNode *nodes, float *query, int k, int *neighbors, float *distances) {
        __shared__ Neighbor best_neighbors[32]; // Adjust size based on max threads per block

        int best_count = 0;
        knn_search(nodes, query, query, 0, 0, k, best_neighbors, best_count);

        for (int i = 0; i < best_count; ++i) {
            neighbors[i] = best_neighbors[i].index;
            distances[i] = best_neighbors[i].distance;
        }
    }

    __device__ void
    radius_search(const KDNode *nodes, const float *points, const float *query, int node_idx, int depth, float radius,
                  int *neighbors, float *distances, int &count) {
        if (node_idx == -1) return;

        const KDNode &node = nodes[node_idx];
        int axis = depth % 3;

        // Compute distance to current point
        float distance = euclidean_distance(query, points + node.index * 3, 3);
        if (distance < radius) {
            neighbors[count] = node.index;
            distances[count] = distance;
            ++count;
        }

        // Determine which side of the split plane to search
        float diff = query[axis] - points[node.index * 3 + axis];
        int near_idx = diff < 0 ? node.left : node.right;
        int far_idx = diff < 0 ? node.right : node.left;

        // Recursively search the nearer side of the split plane
        radius_search(nodes, points, query, near_idx, depth + 1, radius, neighbors, distances, count);

        // If there's potential for the farther side to contain points within the radius, search it as well
        if (fabs(diff) < radius) {
            radius_search(nodes, points, query, far_idx, depth + 1, radius, neighbors, distances, count);
        }
    }

    __global__ void Cuda::radius_query(KDNode *nodes, float *query, float radius, int *neighbors, float *distances) {
        int count = 0;
        radius_search(nodes, query, query, 0, 0, radius, neighbors, distances, count);
    }
}