//
// Created by alex on 06.08.24.
//

#include "KDTreeCuda.h"
#include "KDTreeCuda.cuh"
#include "CudaCommon.h"

namespace Bcg { 
    KDTreeCuda::KDTreeCuda() : index(nullptr){}

    KDTreeCuda::~KDTreeCuda() {
        if (index) {
            cudaFree(index);
        }
    }

    void KDTreeCuda::build(const std::vector<Vector<float, 3>> &positions) {
        size_t num_points = positions.size();

        // Allocate memory for KDTreeCuda nodes
        cudaMalloc(&index, num_points * sizeof(KDNode));

        // Allocate memory for positions on device
        float3 *d_positions;
        cudaMalloc(&d_positions, num_points * sizeof(float3));

        h_positions.resize(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            h_positions[i] = {positions[i][0], positions[i][1], positions[i][2]};
        }
        cudaMemcpy(d_positions, h_positions.data(), num_points * sizeof(float3), cudaMemcpyHostToDevice);

        // Build the KD-tree
        Cuda::build_kdtree(index, d_positions, 0, 0, num_points);

        cudaFree(d_positions);
    }

    QueryResult KDTreeCuda::knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const {
        QueryResult result;

        // Allocate device memory for query results
        float3 *d_query;
        cudaMalloc(&d_query, sizeof(float3));
        cudaMemcpy(d_query, query_point.data(), sizeof(float3), cudaMemcpyHostToDevice);

        int *d_indices;
        float *d_distances;
        cudaMalloc(&d_indices, num_closest * sizeof(int));
        cudaMalloc(&d_distances, num_closest * sizeof(float));

        float3 *d_positions;
        cudaMalloc(&d_positions, h_positions.size() * sizeof(float3));
        cudaMemcpy(d_positions, h_positions.data(), h_positions.size() * sizeof(float3), cudaMemcpyHostToDevice);

        // Perform the kNN query
        Cuda::knn_query(index, d_positions, *d_query, num_closest, d_indices, d_distances);

        // Copy results back to host
        thrust::host_vector<int> h_indices(num_closest);
        thrust::host_vector<float> h_distances(num_closest);
        cudaMemcpy(h_indices.data(), d_indices, num_closest * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_distances.data(), d_distances, num_closest * sizeof(float), cudaMemcpyDeviceToHost);

        // Populate QueryResult
        result.indices.assign(h_indices.begin(), h_indices.end());
        result.distances.assign(h_distances.begin(), h_distances.end());

        // Free device memory
        cudaFree(d_indices);
        cudaFree(d_distances);
        cudaFree(d_positions);
        cudaFree(d_query);

        return result;
    }

    QueryResult KDTreeCuda::radius_query(const Vector<float, 3> &query_point, float radius) const {
        QueryResult result;

        // Allocate device memory for query
        float3 *d_query;
        cudaMalloc(&d_query, sizeof(float3));
        cudaMemcpy(d_query, query_point.data(), sizeof(float3), cudaMemcpyHostToDevice);

        int *d_indices;
        float *d_distances;
        size_t max_results = 1024; // Define a maximum number of results
        cudaMalloc(&d_indices, max_results * sizeof(int));
        cudaMalloc(&d_distances, max_results * sizeof(float));

        float3 *d_positions;
        cudaMalloc(&d_positions, h_positions.size() * sizeof(float3));
        cudaMemcpy(d_positions, h_positions.data(), h_positions.size() * sizeof(float3), cudaMemcpyHostToDevice);

        // Perform the radius query
        Cuda::radius_query(index, d_positions, *d_query, radius, d_indices, d_distances);

        // Copy results back to host
        thrust::host_vector<int> h_indices(max_results);
        thrust::host_vector<float> h_distances(max_results);
        cudaMemcpy(h_indices.data(), d_indices, max_results * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_distances.data(), d_distances, max_results * sizeof(float), cudaMemcpyDeviceToHost);

        // Filter valid results (non-negative indices)
        for (size_t i = 0; i < max_results; ++i) {
            if (h_indices[i] >= 0) {
                result.indices.push_back(h_indices[i]);
                result.distances.push_back(h_distances[i]);
            }
        }

        // Free device memory
        cudaFree(d_indices);
        cudaFree(d_distances);
        cudaFree(d_positions);
        cudaFree(d_query);

        return result;
    }

    QueryResult KDTreeCuda::closest_query(const Vector<float, 3> &query_point) const {
        return knn_query(query_point, 1);
    }
}