//
// Created by alex on 20.08.24.
//

#ifndef ENGINE24_KMEANS_CUH
#define ENGINE24_KMEANS_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/shuffle.h>
#include "bvh_device.cuh"

namespace Bcg::cuda::kmeans {
    inline void check_cuda() {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            assert(cudaSuccess == err);
        }
    }

    template<typename ScalarType, typename VecType, typename IndexType>
    struct KmeansDeviceDataPtr {
        using scalar_type = ScalarType;
        using vec_type = VecType;
        using index_type = IndexType;

        vec_type *d_positions;
        index_type *d_labels;
        scalar_type *d_distances;

        vec_type *d_centroids;
        vec_type *d_new_sums;
        index_type *d_new_cluster_sizes;
        cuda::bvh_device<vec_type> d_bvh;

        unsigned int num_objects;
        unsigned int num_clusters;
    };

    template<typename ScalarType, typename VecType, typename IndexType>
    struct KmeansHostData {
        using scalar_type = ScalarType;
        using vec_type = VecType;
        using index_type = IndexType;

        thrust::host_vector<vec_type> positions;
        thrust::host_vector<index_type> labels;
        thrust::host_vector<scalar_type> distances;

        thrust::host_vector<vec_type> centroids;

        bvh::host_data<vec_type> h_bvh;

        KmeansHostData() = default;

        template<typename InputIterator>
        KmeansHostData(InputIterator begin, InputIterator end,
                       size_t num_clusters) : positions(begin, end),
                                              labels(positions.size(), 0),
                                              distances(positions.size(), 0),
                                              centroids(num_clusters, vec_type::constant(0)) {}
    };

    template<typename ScalarType, typename VecType, typename IndexType>
    struct KmeansDeviceData {
        using scalar_type = ScalarType;
        using vec_type = VecType;
        using index_type = IndexType;

        thrust::device_vector<vec_type> positions;
        thrust::device_vector<index_type> labels;
        thrust::device_vector<scalar_type> distances;

        thrust::device_vector<vec_type> centroids;
        thrust::device_vector<vec_type> new_sums;
        thrust::device_vector<index_type> new_cluster_sizes;
        cuda::bvh_device<vec_type> bvh;

        KmeansDeviceData() = default;

        KmeansDeviceData(const thrust::host_vector<vec_type> &positions, size_t num_clusters) :
                positions(positions),
                labels(positions.size(), 0),
                distances(positions.size(), 0),
                centroids(num_clusters, vec_type::constant(0)),
                new_sums(num_clusters, vec_type::constant(0)),
                new_cluster_sizes(num_clusters, 0) {}
    };


    template<typename ScalarType, typename VecType, typename IndexType>
    void clear_new_clusters(KmeansDeviceData < ScalarType, VecType, IndexType > &data) {
        using vec_type = VecType;

        thrust::fill(data.new_sums.begin(), data.new_sums.end(), vec_type::constant(0));
        thrust::fill(data.new_cluster_sizes.begin(), data.new_cluster_sizes.end(), 0);
    }

    template<typename ScalarType, typename VecType, typename IndexType>
    void assign_clusters(KmeansDeviceData < ScalarType, VecType, IndexType > &data) {
        using vec_type = VecType;
        using index_type = IndexType;

        struct distance_calculator {
            __device__ __host__
            float operator()(const vec_type point, const vec_type object) const

            noexcept {
                return (point.x - object.x) * (point.x - object.x) +
                       (point.y - object.y) * (point.y - object.y) +
                       (point.z - object.z) * (point.z - object.z);
            }
        };

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<index_type>(0),
                         thrust::make_counting_iterator<index_type>(data.num_objects),
        [data] __device__ (index_type idx) {
            const auto &query = data.d_positions[idx];
            const auto best = bvh::query_device(data.bvh, nearest(query), distance_calculator());

            const auto best_cluster = best.first;
            const auto best_distance = best.second;

            data.d_new_labels[idx] = best_cluster;
            data.d_new_distances[idx] = best_distance;

            auto *new_sum = thrust::raw_pointer_cast(data.d_new_sums + best_cluster);

            atomicAdd(&new_sum->x, query.x);
            atomicAdd(&new_sum->y, query.y);
            atomicAdd(&new_sum->z, query.z);
            atomicAdd(thrust::raw_pointer_cast(&data.d_new_cluster_sizes[best_cluster]), 1);
        });

        check_cuda();
    }

    template<typename ScalarType, typename VecType, typename IndexType>
    void update_centroids(KmeansDeviceData < ScalarType, VecType, IndexType > &data) {
        using vec_type = VecType;
        using index_type = IndexType;

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator<index_type>(0),
                         thrust::make_counting_iterator<index_type>(data.num_clusters),
        [data] __device__(index_type idx) {
            const auto cluster_size = data.d_new_cluster_sizes[idx];
            if (cluster_size == 0) {
                return;
            }

            const auto new_sum = data.d_new_sums[idx];
            data.d_centroids[idx] = new_sum / static_cast<float>(cluster_size);
        });

        check_cuda();
    }

    template<typename ScalarType, typename VecType, typename IndexType>
    void initialize_random_centroids(KmeansDeviceData < ScalarType, VecType, IndexType > &data, unsigned int num_clusters) {
        using vec_type = VecType;
        using index_type = IndexType;

        thrust::device_vector<index_type> indices(data.num_objects);
        thrust::sequence(indices.begin(), indices.end());

        thrust::shuffle(indices.begin(), indices.end(), thrust::default_random_engine());

        for (unsigned int i = 0; i < num_clusters; ++i) {
            data.centroids[i] = data.positions[indices[i]];
        }
    }

    template<typename ScalarType, typename VecType, typename IndexType>
    void kmeans_device(KmeansDeviceData < ScalarType, VecType, IndexType > &data, unsigned int max_iterations) {
        initialize_random_centroids(data, data.num_clusters);
        cuda::aabb aabb;
        for (unsigned int i = 0;i < max_iterations;++i) {
            data.bvh = {};
            data.bvh.objects = data.d_positions.data().get();
            data.bvh.num_objects = data.num_clusters;
            bvh::construct_device(data.bvh, aabb);

            clear_new_clusters(data);
            assign_clusters(data);
            update_centroids(data);
        }
    }
}
#endif //ENGINE24_KMEANS_CUH
