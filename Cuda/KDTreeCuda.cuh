//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_KDTREECUDA_CUH
#define ENGINE24_KDTREECUDA_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "KDTreeNode.h"
#include "CudaCommon.h"
#include "MatVec.h"

namespace Bcg::Cuda {
    // Function to build KD-tree (simplified)
    void build_kdtree(KDNode *nodes, float3 *points, int depth, int start, int end);

    void knn_query(KDNode *nodes, const float3 *points, const float3 &query, int k, int *neighbors, float *distances);

    void radius_query(KDNode *nodes, const float3 *points, const float3 &query, float radius, int *neighbors,
                      float *distances);
}

#endif //ENGINE24_KDTREECUDA_CUH
