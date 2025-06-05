//
// Created by alex on 05.06.25.
//

#define FMT_USE_NONTYPE_TEMPLATE_ARGS 0

#include "../BVHCudaNew.h"
#include "Engine.h"
#include "Logger.h"
#include "bvh_device.cuh"
#include "query_device.cuh"

namespace Bcg::cuda {
    using bvh_type = bvh::device_data<vec3>;

    BVHCudaNew::BVHCudaNew(entt::entity entity_id) : entity_id(entity_id) {

    }

    BVHCudaNew::~BVHCudaNew() {

    }

    BVHCudaNew::operator bool() const {
        return Engine::has<bvh_type>(entity_id);
    }

    void BVHCudaNew::build(const std::vector<Vector<float, 3>> &positions) {
        thrust::host_vector<vec3> ps(positions.size());
        cuda::aabb aabb;
        aabb.min = vec3(std::numeric_limits<float>::max());
        aabb.max = vec3(-std::numeric_limits<float>::max());
        for (size_t i = 0; i < positions.size(); ++i) {
            ps[i] = {positions[i][0], positions[i][1], positions[i][2]};
            aabb = merge(aabb, aabb_getter<vec3>()(ps[i]));
        }

        auto &d_bvh = Engine::require<bvh_type>(entity_id);
        d_bvh.objects = ps;

        auto start_time = std::chrono::high_resolution_clock::now();
        bvh::construct_device(d_bvh);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> build_duration = end_time - start_time;
        Log::Info("Build KDTree in {} seconds", build_duration.count());
    }

    [[nodiscard]] QueryResult
    BVHCudaNew::knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const {
        vec3 d_query = {query_point[0], query_point[1], query_point[2]};

        auto &d_bvh = Engine::require<cuda::bvh::device_data<vec3> >(entity_id);
        const cuda::bvh::host_data<vec3> h_bvh = cuda::bvh::get_host_data(d_bvh);

        std::vector<std::pair<unsigned int, float>> results;
        cuda::bvh::query_host(h_bvh, query_knn(d_query, num_closest), distance_calculator<vec3, vec3>(), results);


        QueryResult result(num_closest);
        for (long i = 0; i < num_closest; ++i) {
            result.indices[i] = results[i].first;
            result.distances[i] = results[i].second;
        }
        return result;
    }

    [[nodiscard]] QueryResult BVHCudaNew::radius_query(const Vector<float, 3> &query_point, float radius) const {
        vec3 d_query = {query_point[0], query_point[1], query_point[2]};

        auto &d_bvh = Engine::require<cuda::bvh::device_data<vec3> >(entity_id);
        const cuda::bvh::host_data<vec3> h_bvh = cuda::bvh::get_host_data(d_bvh);

        std::vector<unsigned int> indices_host;
        cuda::bvh::query_host(h_bvh, overlaps_sphere(d_query, radius), indices_host);

        auto num_found = indices_host.size();
        QueryResult result(num_found);
        for (long i = 0; i < num_found; ++i) {
            auto idx = indices_host[i];
            result.indices[i] = idx;
            result.distances[i] = length(d_query - h_bvh.objects[idx]);
        }
        return result;
    }

    [[nodiscard]] QueryResult BVHCudaNew::closest_query(const Vector<float, 3> &query_point) const {
        vec3 d_query = {query_point[0], query_point[1], query_point[2]};

        auto &d_bvh = Engine::require<cuda::bvh::device_data<vec3> >(entity_id);
        const cuda::bvh::host_data<vec3> h_bvh = cuda::bvh::get_host_data(d_bvh);

        auto found = cuda::bvh::query_host(h_bvh, query_nearest(d_query), distance_calculator<vec3, vec3>());

        QueryResult result(1);
        result.indices[0] = found.first;
        result.distances[0] = found.second;
        return result;
    }

    [[nodiscard]] std::vector<QueryResult>
    BVHCudaNew::knn_query_batch(const std::vector<Vector<float, 3>> &query_points, unsigned int num_closest) const {
        unsigned int num_queries = query_points.size();
        thrust::host_vector<vec3> qs(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            qs[i] = {query_points[i][0], query_points[i][1], query_points[i][2]};
        }

        thrust::device_vector<vec3> d_query(qs);
        thrust::device_vector<thrust::pair<unsigned int, float>> results(num_queries * num_closest);
        thrust::device_vector<unsigned int> nums_found(num_queries);

        const auto &d_bvh = Engine::require<cuda::bvh::device_data<vec3> >(entity_id);
        auto d_bvh_ptr = bvh::get_device_ptrs(d_bvh);

        int threadsPerBlock = 256;
        int numBlocks = (num_queries + threadsPerBlock - 1) / threadsPerBlock;
        bvh::query_knn_parallel_fixed_k_kernel<<<numBlocks, threadsPerBlock>>>(
                d_bvh_ptr,
                        d_query.data().get(),
                        num_queries,
                        num_closest,
                        distance_calculator<vec3, vec3>(),
                        results.data().get(),
                        nums_found.data().get()
        );

        thrust::host_vector<thrust::pair<unsigned int, float>> h_results = results;


        std::vector<QueryResult> result(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            unsigned int num_found = nums_found[i];
            result[i].resize(num_found);
            for (unsigned int j = 0; j < num_found; ++j) {
                result[i].indices[j] = h_results[i * num_closest + j].first;
                result[i].distances[j] = h_results[i * num_closest + j].second;
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<QueryResult>
    BVHCudaNew::radius_query_batch(const std::vector<Vector<float, 3>> &query_points, float radius) const {
        unsigned int num_queries = query_points.size();
        thrust::host_vector<sphere> qs(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            qs[i].center = {query_points[i][0], query_points[i][1], query_points[i][2]};
            qs[i].radius = radius;
        }

        thrust::device_vector<sphere> d_query(qs);
        sphere *d_query_ptr = d_query.data().get();
        thrust::device_vector<unsigned int> nums_found(num_queries);

        const auto &d_bvh = Engine::require<cuda::bvh::device_data<vec3> >(entity_id);
        auto d_bvh_ptr = bvh::get_device_ptrs(d_bvh);

        int threadsPerBlock = 256;
        int numBlocks = (num_queries + threadsPerBlock - 1) / threadsPerBlock;
        bvh::query_overlap_count_parallel_kernel<<<numBlocks, threadsPerBlock>>>(
                d_bvh_ptr,
                        d_query_ptr,
                        num_queries,
                        nums_found.data().get()
        );
        unsigned int total_found = thrust::reduce(nums_found.begin(), nums_found.end(), 0,
                                                  thrust::plus<unsigned int>());
        thrust::device_vector<unsigned int> d_results(total_found);
        thrust::device_vector<unsigned int> d_result_offsets(num_queries);

        thrust::exclusive_scan(nums_found.begin(), nums_found.end(), d_result_offsets.begin(), 0);

        bvh::query_overlap_parallel_kernel<<<numBlocks, threadsPerBlock>>>(
                d_bvh_ptr,
                        d_query_ptr,
                        d_result_offsets.data().get(),
                        nums_found.data().get(),
                        num_queries,
                        d_results.data().get()
        );

        thrust::host_vector<unsigned int> h_results = d_results;
        thrust::host_vector<unsigned int> h_num_found = nums_found;


        std::vector<QueryResult> result(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            unsigned int num_found = h_num_found[i];
            result[i].resize(num_found);
            for (unsigned int j = 0; j < num_found; ++j) {
                unsigned int idx = h_results[i * num_found + j];
                result[i].indices[j] = idx;
                result[i].distances[j] = length(qs[i].center - d_bvh.objects[idx]);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<QueryResult>
    BVHCudaNew::closest_query_batch(const std::vector<Vector<float, 3>> &query_points) const {
        unsigned int num_queries = query_points.size();
        thrust::host_vector<vec3> qs(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            qs[i] = {query_points[i][0], query_points[i][1], query_points[i][2]};
        }

        thrust::device_vector<vec3> d_query(qs);
        thrust::device_vector<thrust::pair<unsigned int, float>> results(num_queries);

        const auto &d_bvh = Engine::require<cuda::bvh::device_data<vec3> >(entity_id);
        auto d_bvh_ptr = bvh::get_device_ptrs(d_bvh);

        int threadsPerBlock = 256;
        int numBlocks = (num_queries + threadsPerBlock - 1) / threadsPerBlock;
        bvh::query_nearest_parallel_kernel<<<numBlocks, threadsPerBlock>>>(
                d_bvh_ptr,
                        d_query.data().get(),
                        num_queries,
                        distance_calculator<vec3, vec3>(),
                        results.data().get()
        );

        thrust::host_vector<thrust::pair<unsigned int, float>> h_results = results;


        std::vector<QueryResult> result(num_queries);
        for (size_t i = 0; i < num_queries; ++i) {
            result[i].resize(1);
            result[i].indices[0] = h_results[i].first;
            result[i].distances[0] = h_results[i].second;
        }
        return result;
    }
}