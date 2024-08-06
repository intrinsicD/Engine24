//
// Created by alex on 06.08.24.
//

#include <thrust/sort.h>
#include "KDTreeCuda.cuh"
#include "CudaCommon.h"
#include "thrust/device_ptr.h"

namespace Bcg::Cuda {


    struct ComparePoints {
        int axis;

        __host__ __device__
        explicit ComparePoints(int axis) : axis(axis) {}

        __host__ __device__
        bool operator()(const float3 &a, const float3 &b) const {
            return (&(a.x))[axis] < (&(b.x))[axis];
        }
    };

    __device__ float euclidean_distance(const float3 &a, const float3 &b) {
        float3 diff = {a.x - b.x, a.y + b.y, a.z - b.z};
        float dist = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        return sqrt(dist);
    }

    __global__ void build_kdtree_device(KDNode *nodes, float3 *points, int depth, int start, int end) {
        if (start >= end) return;

        int axis = depth % 3;
        int mid = (start + end) / 2;

        thrust::sort(thrust::device, points + start, points + end, ComparePoints(axis));

        // Set node values
        nodes[mid].split_value = (&(points[mid].x))[axis];
        nodes[mid].index = mid;
        nodes[mid].left = (start < mid) ? mid - 1 : -1;
        nodes[mid].right = (mid < end - 1) ? mid + 1 : -1;
    }

    void build_kdtree(KDNode* nodes, float3 *points, int depth, int start, int end) {
        if (start >= end) return;

        build_kdtree_device<<<1, 1>>>(nodes, points, depth, start, end);
        cudaDeviceSynchronize();

        int mid = (start + end) / 2;
        build_kdtree(nodes, points, depth + 1, start, mid);
        build_kdtree(nodes, points, depth + 1, mid + 1, end);
    }

    struct Neighbor {
        float distance;
        int index;

        __device__ __host__ bool operator<(const Neighbor &other) const {
            return distance < other.distance;
        }
    };

    __device__ void
    knn_search(const KDNode *nodes, const float3 *points, const float3 &query, int node_idx, int depth, int k,
               Neighbor *best_neighbors, int &best_count) {
        if (node_idx == -1) return;

        const KDNode &node = nodes[node_idx];
        int axis = depth % 3;

        // Compute distance to current point
        float distance = euclidean_distance(query, points[node.index]);
        if (best_count < k || distance < best_neighbors[best_count - 1].distance) {
            if (best_count < k) {
                best_neighbors[best_count] = {distance, node.index};
                best_count++;
            } else {
                best_neighbors[best_count - 1] = {distance, node.index};
            }

            // Sort the array to maintain the closest neighbors in the first positions
            for (int i = best_count - 1; i > 0 && best_neighbors[i] < best_neighbors[i - 1]; --i) {
                Neighbor temp = best_neighbors[i];
                best_neighbors[i] = best_neighbors[i - 1];
                best_neighbors[i - 1] = temp;
            }
        }

        // Determine which side of the split plane to search
        float diff = (&(query.x))[axis] - (&(points[node.index].x))[axis];
        int near_idx = diff < 0 ? node.left : node.right;
        int far_idx = diff < 0 ? node.right : node.left;

        // Recursively search the nearer side of the split plane
        knn_search(nodes, points, query, near_idx, depth + 1, k, best_neighbors, best_count);

        // If there's potential for the farther side to contain closer neighbors, search it as well
        if (best_count < k || fabs(diff) < best_neighbors[best_count - 1].distance) {
            knn_search(nodes, points, query, far_idx, depth + 1, k, best_neighbors, best_count);
        }
    }


    __global__ void
    knn_query_device(KDNode *nodes, const float3 *points, const float3 &query, int k, int *neighbors,
                     float *distances) {
        Neighbor best_neighbors[32]; // Adjust size based on max threads per block

        int best_count = 0;
        knn_search(nodes, points, query, 0, 0, k, best_neighbors, best_count);

        for (int i = 0; i < best_count; ++i) {
            neighbors[i] = best_neighbors[i].index;
            distances[i] = best_neighbors[i].distance;
        }
    }

    void knn_query(KDNode *nodes, const float3 *points, const float3 &query, int k, int *neighbors, float *distances) {
        knn_query_device<<<1, 1>>>(nodes, points, query, k, neighbors, distances);
        cudaDeviceSynchronize();
    }

    CUDA_DEVICE void
    radius_search_device(const KDNode *nodes, const float3 *points, const float3 &query, int node_idx, int depth,
                         float radius,
                         int *neighbors, float *distances, int &count) {
        if (node_idx == -1) return;

        const KDNode &node = nodes[node_idx];
        int axis = depth % 3;

        // Compute distance to current point
        float distance = euclidean_distance(query, points[node.index]);
        if (distance < radius) {
            neighbors[count] = node.index;
            distances[count] = distance;
            ++count;
        }

        // Determine which side of the split plane to search
        float diff = (&(query.x))[axis] - (&(points[node.index].x))[axis];
        int near_idx = diff < 0 ? node.left : node.right;
        int far_idx = diff < 0 ? node.right : node.left;

        // Recursively search the nearer side of the split plane
        radius_search_device(nodes, points, query, near_idx, depth + 1, radius, neighbors, distances, count);

        // If there's potential for the farther side to contain points within the radius, search it as well
        if (fabs(diff) < radius) {
            radius_search_device(nodes, points, query, far_idx, depth + 1, radius, neighbors, distances, count);
        }
    }

    __global__ void
    radius_query_device(KDNode *nodes, const float3 *points, const float3 &query, float radius, int *neighbors,
                        float *distances) {
        int count = 0;
        radius_search_device(nodes, points, query, 0, 0, radius, neighbors, distances, count);
    }

    void radius_query(KDNode *nodes, const float3 *points, const float3 &query, float radius, int *neighbors,
                      float *distances) {
        radius_query_device<<<1, 1>>>(nodes, points, query, radius, neighbors, distances);
        cudaDeviceSynchronize();
    }
}