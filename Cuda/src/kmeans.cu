//
// Created by alex on 08.08.24.
//

#include "Cuda/Kmeans.h"
#include "lbvh.cuh"
#include <numeric>
#include <random>
#include "CudaCommon.cuh"
#include "Statistics.h"
#include "PropertyEigenMap.h"
#include "mat_vec_helper.cuh"

namespace Bcg::cuda {
    using hbvh = lbvh<vec3, aabb_getter<vec3>>;
    using dbvh = bvh_device<vec3>;

    struct KmeansDeviceDataPtr {
        thrust::device_ptr<vec3> d_positions;
        thrust::device_ptr<unsigned int> d_labels;
        thrust::device_ptr<float> d_distances;

        thrust::device_ptr<vec3> d_centroids;
        thrust::device_ptr<vec3> d_new_sums;
        thrust::device_ptr<unsigned int> d_new_cluster_sizes;
        dbvh d_bvh;
    };

    struct KmeansDeviceData {
        thrust::device_vector<vec3> positions;
        thrust::device_vector<unsigned int> labels;
        thrust::device_vector<float> distances;

        thrust::device_vector<vec3> centroids;
        thrust::device_vector<vec3> new_sums;
        thrust::device_vector<unsigned int> new_cluster_sizes;
        hbvh bvh;

        void push_new_cluster(const vec3 &new_centroid) {
            centroids.push_back(new_centroid);
            new_sums.push_back(vec3(0));
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

    KMeansResult DeviceToHost(KmeansDeviceData &d_data, thrust::host_vector<vec3> &h_centroids) {
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
            result.centroids[i] = Vector<float, 3>(h_centroids[i][0], h_centroids[i][1], h_centroids[i][2]);
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

    void AssignClusters(dbvh &bvh,
                        thrust::device_ptr<vec3> d_positions,
                        thrust::device_ptr<vec3> d_new_sums,
                        thrust::device_ptr<unsigned int> d_new_cluster_sizes,
                        thrust::device_ptr<unsigned int> d_new_labels,
                        thrust::device_ptr<float> d_new_distances,
                        size_t num_objects) {

        struct distance_calculator {
            __device__ __host__
            float operator()(const vec3 point, const vec3 object) const noexcept {
                return (point[0] - object[0]) * (point[0] - object[0]) +
                       (point[1] - object[1]) * (point[1] - object[1]) +
                       (point[2] - object[2]) * (point[2] - object[2]);
            }
        };

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator<std::uint32_t>(num_objects),
                         [bvh, d_positions, d_new_labels, d_new_distances, d_new_sums, d_new_cluster_sizes] __device__(
                                 std::uint32_t idx) {
                             const vec3 &query = d_positions[idx];
                             const auto best = query_device(bvh, nearest(query), distance_calculator());

                             const auto best_cluster = best.first;
                             const auto best_distance = best.second;

                             d_new_labels[idx] = best_cluster;
                             d_new_distances[idx] = best_distance;

                             vec3 *new_sum = thrust::raw_pointer_cast(d_new_sums + best_cluster);


                             atomicAdd(&((*new_sum)[0]), query[0]);
                             atomicAdd(&((*new_sum)[1]), query[1]);
                             atomicAdd(&((*new_sum)[2]), query[2]);
                             atomicAdd(thrust::raw_pointer_cast(&d_new_cluster_sizes[best_cluster]), 1);
                         }
        );
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
    }

    void UpdateCentroids(thrust::device_ptr<vec3> new_centroids,
                         const thrust::device_ptr<vec3> new_sums,
                         const thrust::device_ptr<unsigned int> &new_cluster_sizes, std::uint32_t k) {
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<std::uint32_t>(0),
                         thrust::make_counting_iterator(k),
                         [new_cluster_sizes, new_sums, new_centroids] __device__(std::uint32_t idx) {
                             float count = ::fmaxf(float(new_cluster_sizes[idx]), 1.0f);
                             const vec3 &new_sum = new_sums[idx];
                             vec3 *new_centroid = thrust::raw_pointer_cast(new_centroids + idx);

                             (*new_centroid)[0] = new_sum[0] / count;
                             (*new_centroid)[1] = new_sum[1] / count;
                             (*new_centroid)[2] = new_sum[2] / count;
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

    KMeansResult KMeans(const std::vector<Vector<float, 3>> &points,
                        const std::vector<Vector<float, 3>> &init_means,
                        unsigned int iterations) {
        auto d_points = cuda_helpers::to_device_blit(points);
        auto d_means = cuda_helpers::to_device_blit(init_means);
        auto h_means = thrust::host_vector<vec3>(d_means);

        int k = init_means.size();
        int num_objects = points.size();

        KmeansDeviceData d_data = {d_points,
                                   thrust::device_vector<unsigned int>(num_objects),
                                   thrust::device_vector<unsigned int>(num_objects),
                                   d_means,
                                   thrust::device_vector<vec3>(k),
                                   thrust::device_vector<unsigned int>(k),
                                   hbvh(h_means.begin(), h_means.end(), true)
        };

        KmeansDeviceDataPtr d_ptrs = d_data.get_ptrs();

        for (size_t i = 0; i < iterations; ++i) {
            d_data.bvh.clear();
            d_data.bvh.assign_device(d_data.centroids);

            d_ptrs.d_bvh = d_data.bvh.get_device_repr();

            thrust::fill(d_data.new_sums.begin(), d_data.new_sums.end(), vec3(0));
            thrust::fill(d_data.new_cluster_sizes.begin(), d_data.new_cluster_sizes.end(), 0);

            AssignClusters(d_ptrs.d_bvh, d_ptrs.d_positions, d_ptrs.d_new_sums, d_ptrs.d_new_cluster_sizes,
                           d_ptrs.d_labels, d_ptrs.d_distances, num_objects);

            UpdateCentroids(d_ptrs.d_centroids, d_ptrs.d_new_sums, d_ptrs.d_new_cluster_sizes, k);
        }

        return DeviceToHost(d_data, h_means);
    }

    vec3 GetMostDistantPoint(KmeansDeviceData &d_data) {
        auto max_dist_iter = thrust::max_element(d_data.distances.begin(), d_data.distances.end());
        auto max_dist_index = max_dist_iter - d_data.distances.begin();
        return d_data.positions[max_dist_index];
    }

    KMeansResult
    HierarchicalKMeans(const std::vector<Vector<float, 3>> &points, unsigned int k, unsigned int iterations) {
        int current_k = 1;
        int num_objects = points.size();

        const Vector<float, 3> mean = Mean(points);
        thrust::host_vector<vec3> h_centroids;
        thrust::host_vector<float> h_distances(num_objects);
        h_centroids.push_back({mean[0], mean[1], mean[2]});

        thrust::host_vector<vec3> h_positions(num_objects);
        for (size_t i = 0; i < num_objects; ++i) {
            h_positions[i] = {points[i][0], points[i][1], points[i][2]};
            Vector<float, 3> diff = points[i] - mean;
            h_distances[i] = glm::dot(diff, diff);
        }

        KmeansDeviceData d_data = {thrust::device_vector<vec3>(h_positions),
                                   thrust::device_vector<unsigned int>(num_objects),
                                   thrust::device_vector<unsigned int>(num_objects),
                                   h_centroids,
                                   thrust::device_vector<vec3>(k),
                                   thrust::device_vector<unsigned int>(k),
                                   hbvh(h_centroids.begin(), h_centroids.end(), true)
        };
        d_data.distances = h_distances;

        d_data.push_new_cluster(GetMostDistantPoint(d_data));
        current_k = d_data.centroids.size();

        while (current_k < k) {
            for (size_t i = 0; i < iterations; ++i) {
                d_data.bvh.clear();
                d_data.bvh.assign_device(d_data.centroids);

                KmeansDeviceDataPtr d_ptrs = d_data.get_ptrs();

                thrust::fill(d_data.new_sums.begin(), d_data.new_sums.end(), vec3(0));
                thrust::fill(d_data.new_cluster_sizes.begin(), d_data.new_cluster_sizes.end(), 0);

                AssignClusters(d_ptrs.d_bvh, d_ptrs.d_positions, d_ptrs.d_new_sums, d_ptrs.d_new_cluster_sizes,
                               d_ptrs.d_labels, d_ptrs.d_distances, num_objects);

                UpdateCentroids(d_ptrs.d_centroids, d_ptrs.d_new_sums, d_ptrs.d_new_cluster_sizes, current_k);
            }

            d_data.push_new_cluster(GetMostDistantPoint(d_data));
            current_k = d_data.centroids.size();
        }

        h_centroids.resize(current_k);
        return DeviceToHost(d_data, h_centroids);
    }
}