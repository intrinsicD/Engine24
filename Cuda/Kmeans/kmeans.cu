//
// Created by alex on 08.08.24.
//

#include "Kmeans.h"
#include "lbvh.cuh"
#include <numeric>
#include <random>
#include "CudaCommon.cuh"
#include "Statistics.h"
#include "PropertyEigenMap.h"

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

    struct KmeansDeviceDataPtr {
        thrust::device_ptr<float4> d_positions;
        thrust::device_ptr<unsigned int> d_labels;
        thrust::device_ptr<float> d_distances;

        thrust::device_ptr<float4> d_centroids;
        thrust::device_ptr<float4> d_new_sums;
        thrust::device_ptr<unsigned int> d_new_cluster_sizes;
        dbvh d_bvh;
    };

    struct KmeansDeviceData {
        thrust::device_vector<float4> positions;
        thrust::device_vector<unsigned int> labels;
        thrust::device_vector<float> distances;

        thrust::device_vector<float4> centroids;
        thrust::device_vector<float4> new_sums;
        thrust::device_vector<unsigned int> new_cluster_sizes;
        hbvh bvh;

        void push_new_cluster(const float4 &new_centroid) {
            centroids.push_back(new_centroid);
            new_sums.push_back({0, 0, 0, 0});
            new_cluster_sizes.push_back(0);
        }

        KmeansDeviceDataPtr get_ptrs() {
            return {positions.data(),
                    labels.data(),
                    distances.data(),
                    centroids.data(),
                    new_sums.data(),
                    new_cluster_sizes.data(),
                    bvh.get_device_repr()};
        }
    };

    KmeansDeviceData SetupKMeansDeviceData(const thrust::host_vector<float4> &h_centroids,
                                           const thrust::host_vector<float4> h_positions) {
        auto num_objects = h_positions.size();
        auto k = h_centroids.size();
        return {h_positions,
                thrust::device_vector<unsigned int>(num_objects),
                thrust::device_vector<unsigned int>(num_objects),
                h_centroids,
                thrust::device_vector<float4>(k),
                thrust::device_vector<unsigned int>(k),
                hbvh(h_centroids.begin(), h_centroids.end(), true)
        };
    }


    struct DistanceSumMax {
        float sum;
        float max_distance;
        int max_index;

        __host__ __device__
        DistanceSumMax() : sum(0.0f), max_distance(0.0f), max_index(-1) {}

        __host__ __device__
        DistanceSumMax(float s, float max_dist, int max_idx) : sum(s), max_distance(max_dist), max_index(max_idx) {}
    };

    struct DistanceSumMaxOp {
        __host__ __device__
        DistanceSumMax operator()(const DistanceSumMax &a, const DistanceSumMax &b) const {
            float total_sum = a.sum + b.sum;
            if (a.max_distance > b.max_distance) {
                return {total_sum, a.max_distance, a.max_index};
            } else {
                return {total_sum, b.max_distance, b.max_index};
            }
        }
    };

    struct CreateDistanceSumMax {
        __host__ __device__
        DistanceSumMax operator()(const thrust::tuple<float, int> &t) const {
            float distance = thrust::get<0>(t);
            int index = thrust::get<1>(t);
            return {distance, distance, index};
        }
    };

    KMeansResult DeviceToHost(KmeansDeviceData &d_data, thrust::host_vector<float4> &h_centroids) {
        // Combine distance sum and max distance finding
        int num_objects = d_data.positions.size();
        int k = d_data.centroids.size();

        thrust::device_vector<int> indices(num_objects);
        thrust::sequence(indices.begin(), indices.end());

        auto iter_begin = thrust::make_zip_iterator(thrust::make_tuple(d_data.distances.begin(), indices.begin()));
        auto iter_end = thrust::make_zip_iterator(thrust::make_tuple(d_data.distances.end(), indices.end()));

        DistanceSumMax init;
        DistanceSumMax result_data = thrust::transform_reduce(iter_begin, iter_end,
                                                              CreateDistanceSumMax(),
                                                              init,
                                                              DistanceSumMaxOp());

        // Prepare the KMeansResult
        KMeansResult result;

        // Copy centroids to result
        thrust::copy(d_data.centroids.begin(), d_data.centroids.end(), h_centroids.begin());
        result.centroids.resize(k);
        for (size_t i = 0; i < k; ++i) {
            result.centroids[i] = Vector<float, 3>(h_centroids[i].x, h_centroids[i].y, h_centroids[i].z);
        }

        // Copy labels and distances to result
        result.labels.resize(num_objects);
        result.distances.resize(num_objects);

        thrust::copy(d_data.labels.begin(), d_data.labels.end(), result.labels.begin());
        thrust::copy(d_data.distances.begin(), d_data.distances.end(), result.distances.begin());

        result.error = result_data.sum;
        result.max_dist_index = result_data.max_index;

        return result;
    }

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

        KmeansDeviceData d_data = SetupKMeansDeviceData(h_centroids, h_positions);
        KmeansDeviceDataPtr d_ptrs = d_data.get_ptrs();

        for (size_t i = 0; i < iterations; ++i) {
            d_data.bvh.clear();
            d_data.bvh.assign_device(d_data.centroids);

            d_ptrs.d_bvh = d_data.bvh.get_device_repr();

            thrust::fill(d_data.new_sums.begin(), d_data.new_sums.end(), float4(0, 0, 0, 0));
            thrust::fill(d_data.new_cluster_sizes.begin(), d_data.new_cluster_sizes.end(), 0);

            assign_clusters(d_ptrs.d_bvh, d_ptrs.d_positions, d_ptrs.d_new_sums, d_ptrs.d_new_cluster_sizes,
                            d_ptrs.d_labels, d_ptrs.d_distances, num_objects);

            update_centroids(d_ptrs.d_centroids, d_ptrs.d_new_sums, d_ptrs.d_new_cluster_sizes, k);
        }

        return DeviceToHost(d_data, h_centroids);
    }

    KMeansResult
    HierarchicalKMeans(const std::vector<Vector<float, 3>> &points, unsigned int k, unsigned int iterations) {
        int current_k = 1;
        int num_objects = points.size();

        const Vector<float, 3> mean = Mean(points);
        thrust::host_vector<float4> h_centroids;
        thrust::host_vector<float> h_distances(num_objects);
        h_centroids.push_back({mean.x(), mean.y(), mean.z(), 1.0f});

        thrust::host_vector<float4> h_positions(num_objects);
        for (size_t i = 0; i < num_objects; ++i) {
            h_positions[i] = {points[i].x(), points[i].y(), points[i].z(), 1.0f};
            h_distances[i] = (points[i] - mean).squaredNorm();
        }

        KmeansDeviceData d_data = SetupKMeansDeviceData(h_centroids, h_positions);
        d_data.distances = h_distances;


        auto max_dist_iter = thrust::max_element(d_data.distances.begin(), d_data.distances.end());
        auto max_dist_index = max_dist_iter - d_data.distances.begin();
        d_data.push_new_cluster(d_data.positions[max_dist_index]);
        current_k = d_data.centroids.size();

        while (current_k < k) {
            for (size_t i = 0; i < iterations; ++i) {
                d_data.bvh.clear();
                d_data.bvh.assign_device(d_data.centroids);

                KmeansDeviceDataPtr d_ptrs = d_data.get_ptrs();

                //d_ptrs.d_bvh = d_data.bvh.get_device_repr();

                thrust::fill(d_data.new_sums.begin(), d_data.new_sums.end(), float4(0, 0, 0, 0));
                thrust::fill(d_data.new_cluster_sizes.begin(), d_data.new_cluster_sizes.end(), 0);

                assign_clusters(d_ptrs.d_bvh, d_ptrs.d_positions, d_ptrs.d_new_sums, d_ptrs.d_new_cluster_sizes,
                                d_ptrs.d_labels, d_ptrs.d_distances, num_objects);

                update_centroids(d_ptrs.d_centroids, d_ptrs.d_new_sums, d_ptrs.d_new_cluster_sizes, current_k);
            }
            auto max_dist_iter = thrust::max_element(d_data.distances.begin(), d_data.distances.end());
            auto max_dist_index = max_dist_iter - d_data.distances.begin();
            d_data.push_new_cluster(d_data.positions[max_dist_index]);
            current_k = d_data.centroids.size();
        }

        h_centroids.resize(current_k);
        return DeviceToHost(d_data, h_centroids);
    }
}