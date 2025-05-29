//
// Created by alex on 20.08.24.
//

#ifndef ENGINE24_KMEANS_CUH
#define ENGINE24_KMEANS_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
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

        thrust::host_vector <vec_type> positions;
        thrust::host_vector <index_type> labels;
        thrust::host_vector <scalar_type> distances;

        thrust::host_vector <vec_type> centroids;

        KmeansHostData() = default;

        template<typename InputIterator>
        KmeansHostData(InputIterator begin, InputIterator end,
                       size_t num_clusters) : positions(begin, end),
                                              labels(positions.size(), 0),
                                              distances(positions.size(), 0),
                                              centroids(num_clusters, vec_type::constant(0)) {}
    };

    template<typename ScalarType, typename VecType, typename IndexType, typename AABBGetter>
    struct KmeansDeviceData {
        using scalar_type = ScalarType;
        using vec_type = VecType;
        using index_type = IndexType;

        KmeansDeviceData() = default;

        KmeansDeviceData(const thrust::host_vector <vec_type> &positions, size_t num_clusters) :
                positions(positions),
                labels(positions.size(), 0),
                distances(positions.size(), 0),
                centroids(num_clusters, vec_type::constant(0)),
                new_sums(num_clusters, vec_type::constant(0)),
                new_cluster_sizes(num_clusters, 0) {
            bvh = lbvh<vec_type, AABBGetter>(positions.begin(), positions.end());
        }
    };

    KmeansDeviceData(const thrust::host_vector <vec_type> &positions,
                     const thrust::host_vector <vec_type> &init_clusters) :
            positions(positions),
            labels(positions.size(), 0),
            distances(positions.size(), 0),
            centroids(init_clusters.size(), vec_type::constant(0)),
            new_sums(init_clusters.size(), vec_type::constant(0)),
            new_cluster_sizes(init_clusters.size(), 0) {}


    thrust::device_vector <vec_type> positions;
    thrust::device_vector <index_type> labels;
    thrust::device_vector <scalar_type> distances;

    thrust::device_vector <vec_type> centroids;
    thrust::device_vector <vec_type> new_sums;
    thrust::device_vector <index_type> new_cluster_sizes;
    lbvh<vec_type, AABBGetter> bvh;
};

template<typename ScalarType, typename VecType, typename IndexType>
KmeansDeviceDataPtr get_device_ptrs(KmeansDeviceData &data) {
    return {data.positions.data().get(),
            data.labels.data().get(),
            data.distances.data().get(),
            data.centroids.data().get(),
            data.new_sums.data().get(),
            data.new_cluster_sizes.data().get(),
            data.bvh.get_device_repr(),
            static_cast<unsigned int>(data.positions.size()),
            static_cast<unsigned int>(data.centroids.size())};
}

template<typename ScalarType, typename VecType, typename IndexType>
KmeansDeviceData get_device_data(KmeansDeviceDataPtr &ptrs) {
    return {thrust::device_vector<VecType>(ptrs.d_positions, ptrs.d_positions + ptrs.num_objects),
            thrust::device_vector<IndexType>(ptrs.d_labels, ptrs.d_labels + ptrs.num_objects),
            thrust::device_vector<ScalarType>(ptrs.d_distances, ptrs.d_distances + ptrs.num_objects),
            thrust::device_vector<VecType>(ptrs.d_centroids, ptrs.d_centroids + ptrs.num_clusters),
            thrust::device_vector<VecType>(ptrs.d_new_sums, ptrs.d_new_sums + ptrs.num_clusters),
            thrust::device_vector<IndexType>(ptrs.d_new_cluster_sizes,
                                             ptrs.d_new_cluster_sizes + ptrs.num_clusters),
            lbvh<vec_type, AABBGetter>(ptrs.d_bvh),
    };
}

template<typename ScalarType, typename VecType, typename IndexType>
void push_new_cluster(KmeansDeviceDataPtr &data, const VecType &new_centroid) {
    using vec_type = VecType;

    data.centroids.push_back(new_centroid);
    data.new_sums.push_back(vec_type::constant(0));
    data.new_cluster_sizes.push_back(0);
}

template<typename ScalarType, typename VecType, typename IndexType>
void clear_new_clusters(KmeansDeviceDataPtr &data) {
    using vec_type = VecType;

    thrust::fill(data.new_sums.begin(), data.new_sums.end(), vec_type::constant(0));
    thrust::fill(data.new_cluster_sizes.begin(), data.new_cluster_sizes.end(), 0);
}

template<typename ScalarType, typename VecType, typename IndexType>
void assign_clusters(KmeansDeviceDataPtr &data) {
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
    [data] __device__(index_type
    idx) {
        const auto &query = data.d_positions[idx];
        const auto best = query_device(data.bvh, nearest(query), distance_calculator());

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
void update_centroids(KmeansDeviceDataPtr &data) {
    using vec_type = VecType;
    using index_type = IndexType;

    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<index_type>(0),
                     thrust::make_counting_iterator<index_type>(data.num_clusters),
    [data] __device__(index_type
    idx) {
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
KmeansDeviceDataPtr kmeans_device(KmeansDeviceDataPtr &data, unsigned int max_iterations) {
    for (unsigned int i = 0; i < max_iterations; ++i) {
        data.d_bvh->clear();
        data.d_bvh->assign_device(data.d_centroids, data.num_clusters);

        clear_new_clusters(data);
        assign_clusters(data);
        update_centroids(data);
    }
    return data;
}

#endif //ENGINE24_KMEANS_CUH
