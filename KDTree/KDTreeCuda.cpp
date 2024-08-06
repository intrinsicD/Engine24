//
// Created by alex on 06.08.24.
//

#include "KDTreeCuda.h"
#include "KDTreeCuda.cuh"
#include "CudaCommon.h"

namespace Bcg{
    KDTree::KDTree() :index(nullptr){

    }

    KDTree::~KDTree(){
        if (index) {
            cudaFree(index);
        }
    }

    void KDTree::build(const std::vector<Vector<float, 3>> &positions){
        size_t num_points = positions.size();

        // Allocate memory for KDTree nodes
        cudaMalloc(&index, num_points * sizeof(KDNode));

        // Flatten positions vector for CUDA
        float *d_positions;
        cudaMalloc(&d_positions, num_points * 3 * sizeof(float));
        thrust::host_vector<float> h_positions(num_points * 3);
        for (size_t i = 0; i < num_points; ++i) {
            h_positions[i * 3] = positions[i][0];
            h_positions[i * 3 + 1] = positions[i][1];
            h_positions[i * 3 + 2] = positions[i][2];
        }
        cudaMemcpy(d_positions, h_positions.data(), num_points * 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Build the KD-tree
        Cuda::build_kdtree<<<1, 1>>>(index, d_positions, 0, 0, num_points);

        // Free temporary positions memory
        cudaFree(d_positions);
    }

    QueryResult KDTree::knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const{
        QueryResult result;

        // Allocate device memory for query results
        float *d_query;
        cudaMalloc(&d_query, 3 * sizeof(float));
        cudaMemcpy(d_query, query_point.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        int *d_indices;
        float *d_distances;
        cudaMalloc(&d_indices, num_closest * sizeof(int));
        cudaMalloc(&d_distances, num_closest * sizeof(float));

        // Perform the kNN query
        Cuda::knn_query<<<1, 1>>>(index, d_query, num_closest, d_indices, d_distances);

        // Copy results back to host
        thrust::host_vector<int> h_indices(num_closest);
        thrust::host_vector<float> h_distances(num_closest);
        cudaMemcpy(h_indices.data(), d_indices, num_closest * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_distances.data(), d_distances, num_closest * sizeof(float), cudaMemcpyDeviceToHost);

        // Populate QueryResult
        result.indices.assign(h_indices.begin(), h_indices.end());
        result.distances.assign(h_distances.begin(), h_distances.end());

        // Free device memory
        cudaFree(d_query);
        cudaFree(d_indices);
        cudaFree(d_distances);

        return result;
    }

    QueryResult KDTree::radius_query(const Vector<float, 3> &query_point, float radius) const{
        QueryResult result;

        // Allocate device memory for query
        float *d_query;
        cudaMalloc(&d_query, 3 * sizeof(float));
        cudaMemcpy(d_query, query_point.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

        int *d_indices;
        float *d_distances;
        size_t max_results = 1024; // Define a maximum number of results
        cudaMalloc(&d_indices, max_results * sizeof(int));
        cudaMalloc(&d_distances, max_results * sizeof(float));

        // Perform the radius query
        Cuda::radius_query<<<1, 1>>>(index, d_query, radius, d_indices, d_distances);

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
        cudaFree(d_query);
        cudaFree(d_indices);
        cudaFree(d_distances);

        return result;
    }

    QueryResult KDTree::closest_query(const Vector<float, 3> &query_point) const{
        return knn_query(query_point, 1);
    }
}