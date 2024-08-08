//
// Created by alex on 08.08.24.
//

#include "Kmeans.h"
#include "lbvh.cuh"
#include <numeric>
#include <random>
#include "CudaCommon.cuh"

namespace Bcg {
    using hbvh = lbvh::bvh<float, float4, lbvh::aabb_getter>;
    using dbvh = lbvh::cbvh_device<float, float4>;

    struct distance_calculator {
        __device__ __host__
        float operator()(const float4 point, const float4 object) const noexcept {
            return (point.x - object.x) * (point.x - object.x) +
                   (point.y - object.y) * (point.y - object.y) +
                   (point.z - object.z) * (point.z - object.z);
        }
    };

    __global__ void assign_clusters(const dbvh &bvh, float3 *new_sums,
                                    unsigned int *new_cluster_sizes,
                                    unsigned int *new_labels,
                                    float *new_distances) {
        //determine idx of thread
        const auto idx = threadIdx.x + blockIdx.x * blockDim.x;

        const auto d_query = bvh.objects[idx];
        const auto best = lbvh::query_device(bvh, lbvh::nearest(d_query), distance_calculator());

        const auto best_cluster = best.first;
        const auto best_distance = best.second;

        new_labels[idx] = best_cluster;
        new_distances[idx] = best_distance;

        atomicAdd(&new_sums[best_cluster].x, d_query.x);
        atomicAdd(&new_sums[best_cluster].y, d_query.y);
        atomicAdd(&new_sums[best_cluster].z, d_query.z);
        atomicAdd(&new_cluster_sizes[best_cluster], 1);
    }

    template<typename T>
    __global__ void reset_to(T *array, size_t size, T value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            array[idx] = value;
        }
    }

    void reset(float3 *d_array, int numElements) {
        // Define the number of threads per block and the number of blocks
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements * 3 + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the kernel to reset the array
        reset_to<<<blocksPerGrid, threadsPerBlock>>>(reinterpret_cast<float *>(d_array), numElements, 0.0f);

        // Wait for the kernel to complete
        cudaDeviceSynchronize();
    }

    void reset(unsigned int *d_array, int numElements) {
        // Define the number of threads per block and the number of blocks
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the kernel to reset the array
        reset_to<<<blocksPerGrid, threadsPerBlock>>>(d_array, numElements, (unsigned int) 0);

        // Wait for the kernel to complete
        cudaDeviceSynchronize();
    }

    __global__ void
    update_centroids(float3 *new_centroids, const float3 *new_sums, const unsigned int *new_cluster_sizes) {
        const unsigned int idx = threadIdx.x;
        const float count = ::fmaxf(float(new_cluster_sizes[idx]), 1.0f);
        new_centroids[idx].x = new_sums[idx].x / count;
        new_centroids[idx].y = new_sums[idx].y / count;
        new_centroids[idx].z = new_sums[idx].z / count;
    }

    template<typename Point>
    std::vector<Point> RandomCentroids(const std::vector<Point> &positions, unsigned int k) {
        std::vector<unsigned int> indices(positions.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 rng(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), rng);

        std::vector<Point> centroids(k);
        for (size_t i = 0; i < k; ++i) {
            centroids[i] = positions[indices[i]];
        }
        return centroids;
    }

    KMeansResult KMeans(const std::vector<Vector<float, 3>> &positions, unsigned int k, unsigned int iterations) {
        //TODO dimension mismatch between positions (float3) and float4... figure out what to do, becahsue float4 could be faster.
        std::vector<float4> ps(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            ps[i] = {positions[i].x(), positions[i].y(), positions[i].z(), 1.0f};
        }
        auto centroids = RandomCentroids(ps, k);

        int num_objects = positions.size();
        int threads_assign = 256;
        int blocks_assign = (num_objects + threads_assign - 1) / threads_assign;

        int threads_clusters = 256;
        int blocks_clusters = (k + threads_clusters - 1) / threads_clusters;
        int blocks_cluster_coords = (k * 3 + threads_clusters - 1) / threads_clusters;

        float3 *d_centroids;
        cudaMalloc(&d_centroids, k * sizeof(float3));
        cudaMemcpy(d_centroids, centroids.data(), k * sizeof(float3), cudaMemcpyHostToDevice);

        float3 *d_new_sums;
        cudaMalloc(&d_new_sums, k * sizeof(float3));

        unsigned int *d_new_cluster_sizes;
        cudaMalloc(&d_new_cluster_sizes, k * sizeof(unsigned int));

        unsigned int *d_labels;
        cudaMalloc(&d_labels, positions.size() * sizeof(unsigned int));

        float *d_distances;
        cudaMalloc(&d_distances, positions.size() * sizeof(float));

        for (size_t i = 0; i < iterations; ++i) {
            const auto bvh = hbvh(centroids.begin(), centroids.end(), false);

            reset_to<<<blocks_cluster_coords, threads_clusters>>>(reinterpret_cast<float *>(d_new_sums), k, 0.0f);
            cudaDeviceSynchronize();
            reset_to<<<blocks_clusters, threads_clusters>>>(d_new_cluster_sizes, k, (unsigned int) 0);
            cudaDeviceSynchronize();

            assign_clusters<<<blocks_assign, threads_assign>>>(bvh.get_device_repr(), d_new_sums, d_new_cluster_sizes,
                                                               d_labels, d_distances);
            CudaCheckErrorAndSync(__func__);
            update_centroids<<<blocks_clusters, threads_clusters>>>(d_centroids, d_new_sums, d_new_cluster_sizes);
            CudaCheckErrorAndSync(__func__);
        }

        KMeansResult result;
        result.centroids.resize(k);
        cudaMemcpy(result.centroids.data(), d_centroids, k * sizeof(float3), cudaMemcpyDeviceToHost);

        result.labels.resize(positions.size());
        cudaMemcpy(result.labels.data(), d_labels, positions.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        result.distances.resize(positions.size());
        cudaMemcpy(result.distances.data(), d_distances, positions.size() * sizeof(float), cudaMemcpyDeviceToHost);

        result.error = thrust::reduce(d_distances, d_distances + positions.size(), 0.0f, thrust::plus<float>());

        cudaFree(d_centroids);
        cudaFree(d_new_sums);
        cudaFree(d_new_cluster_sizes);
        cudaFree(d_labels);
        cudaFree(d_distances);

        return result;
    }
}