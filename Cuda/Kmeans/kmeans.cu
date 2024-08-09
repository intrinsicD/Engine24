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
    using dbvh = lbvh::bvh_device<float, float4>;

    struct distance_calculator {
        __device__ __host__
        float operator()(const float4 point, const float4 object) const noexcept {
            return (point.x - object.x) * (point.x - object.x) +
                   (point.y - object.y) * (point.y - object.y) +
                   (point.z - object.z) * (point.z - object.z);
        }
    };

    void assign_clusters(dbvh &bvh,
                         thrust::device_ptr<float4> d_positions,
                         thrust::device_ptr<float4> d_new_sums,
                         thrust::device_ptr<unsigned int> d_new_cluster_sizes,
                         thrust::device_ptr<unsigned int> d_new_labels,
                         thrust::device_ptr<float> d_new_distances,
                         size_t num_objects) {

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(num_objects),
                         [bvh, d_positions, d_new_labels, d_new_distances, d_new_sums, d_new_cluster_sizes] __device__(
                                 std::uint32_t idx) {
                             const float4 &query = d_positions[idx];
                             const auto best = lbvh::query_device(bvh, lbvh::nearest(query), distance_calculator());

                             const auto best_cluster = best.first;
                             const auto best_distance = best.second;

                             d_new_labels[idx] = best_cluster;
                             d_new_distances[idx] = best_distance;

                             float4 *new_sum = thrust::raw_pointer_cast(d_new_sums + best_cluster);

                             atomicAdd(&new_sum->x, query.x);
                             atomicAdd(&new_sum->y, query.y);
                             atomicAdd(&new_sum->z, query.z);
                             atomicAdd(thrust::raw_pointer_cast(&d_new_cluster_sizes[best_cluster]), 1);
                         }
        );
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
    }

    void update_centroids(thrust::device_ptr<float4> new_centroids,
                          const thrust::device_ptr<float4> new_sums,
                          const thrust::device_ptr<unsigned int> &new_cluster_sizes, std::uint32_t k) {
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator(k),
                         [new_cluster_sizes, new_sums, new_centroids] __device__(std::uint32_t idx) {
                             float count = ::fmaxf(float(new_cluster_sizes[idx]), 1.0f);
                             const float4 &new_sum = new_sums[idx];
                             float4 *new_centroid = thrust::raw_pointer_cast(new_centroids + idx);

                             new_centroid->x = new_sum.x / count;
                             new_centroid->y = new_sum.y / count;
                             new_centroid->z = new_sum.z / count;
                             new_centroid->w = new_sum.w / count;
                         });
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
    }

    template<typename Point>
    inline std::vector<Point> RandomCentroids(const std::vector<Point> &positions, unsigned int k) {
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
        auto centroids = RandomCentroids(positions, k);
        return KMeans(positions, centroids, iterations);
    }

    KMeansResult KMeans(const std::vector<Vector<float, 3>> &points, const std::vector<Vector<float, 3>> &init_means,
                        unsigned int iterations) {
        int k = init_means.size();
        int num_objects = points.size();

        thrust::host_vector<float4> h_centroids(k);
        for (size_t i = 0; i < k; ++i) {
            h_centroids[i] = {init_means[i].x(), init_means[i].y(), init_means[i].z(), 1.0f};
        }
        thrust::host_vector<float4> h_positions(num_objects);
        for (size_t i = 0; i < num_objects; ++i) {
            h_positions[i] = {points[i].x(), points[i].y(), points[i].z(), 1.0f};
        }

        thrust::device_vector<float4> positions = h_positions;
        thrust::device_vector<unsigned int> labels(num_objects);
        thrust::device_vector<float> distances(num_objects);

        thrust::device_vector<float4> centroids = h_centroids;
        thrust::device_vector<float4> new_sums(k);
        thrust::device_vector<unsigned int> new_cluster_sizes(k);

        thrust::device_ptr<float4> d_positions = positions.data();
        thrust::device_ptr<unsigned int> d_labels = labels.data();
        thrust::device_ptr<float> d_distances = distances.data();

        thrust::device_ptr<float4> d_centroids = centroids.data();
        thrust::device_ptr<float4> d_new_sums = new_sums.data();
        thrust::device_ptr<unsigned int> d_new_cluster_sizes = new_cluster_sizes.data();

        hbvh bvh = hbvh(h_centroids.begin(), h_centroids.end(), true);
        for (size_t i = 0; i < iterations; ++i) {
            bvh.clear();
            bvh.assign_device(centroids);

            auto d_bvh = bvh.get_device_repr();

            thrust::fill(new_sums.begin(), new_sums.end(), float4(0, 0, 0, 0));
            thrust::fill(new_cluster_sizes.begin(), new_cluster_sizes.end(), 0);

            assign_clusters(d_bvh, d_positions, d_new_sums, d_new_cluster_sizes, d_labels, d_distances, num_objects);

            update_centroids(d_centroids, d_new_sums, d_new_cluster_sizes, k);
        }

        KMeansResult result;
        result.centroids.resize(k);
        h_centroids = thrust::host_vector<float4>(centroids);
        for (size_t i = 0; i < k; ++i) {
            result.centroids[i] = Vector<float, 3>(h_centroids[i].x, h_centroids[i].y, h_centroids[i].z);
        }

        result.labels.resize(num_objects);
        result.distances.resize(num_objects);
        for (size_t i = 0; i < num_objects; ++i) {
            result.labels[i] = labels[i];
            result.distances[i] = distances[i];
        }

        result.error = thrust::reduce(thrust::device, distances.begin(), distances.end(), 0.0f);

        return result;
    }
}