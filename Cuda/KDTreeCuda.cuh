//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_KDTREECUDA_CUH
#define ENGINE24_KDTREECUDA_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace Bcg::Cuda {
    // KD-tree node definition
    struct KDNode {
        float split_value;
        int left, right, index;
    };

    // Function to build KD-tree (simplified)
    __global__ void build_kdtree(KDNode *nodes, float *points, int depth, int start, int end);

    __global__ void knn_query(KDNode *nodes, float *points, float *query, int k, int *neighbors);

    __global__ void
    radius_query(KDNode *nodes, float *points, float *query, float radius, thrust::device_vector<int> &results);

}

#endif //ENGINE24_KDTREECUDA_CUH
