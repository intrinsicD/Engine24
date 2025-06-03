//
// Created by alex on 08.08.24.
//
#define FMT_USE_NONTYPE_TEMPLATE_ARGS 0

#include "Cuda/KDTreeCuda.h"
#include "lbvh.cuh"
#include "query_device.cuh"
#include "Engine.h"
#include "Logger.h"

#include <vector>
#include <numeric>

#include "kmeans.cuh"

namespace Bcg::cuda {
    ;

    KDTreeCuda::KDTreeCuda(entt::entity entity_id) : entity_id(entity_id) {
    }

    KDTreeCuda::~KDTreeCuda() = default;

    KDTreeCuda::operator bool() const {
        return Engine::has<lbvh<vec3, aabb_getter<vec3> > >(entity_id);
    }

    void KDTreeCuda::build(const std::vector<Vector<float, 3> > &positions) {
        thrust::host_vector<vec3> ps(positions.size());
        cuda::aabb aabb;
        aabb.min = vec3(std::numeric_limits<float>::max());
        aabb.max = vec3(-std::numeric_limits<float>::max());
        for (size_t i = 0; i < positions.size(); ++i) {
            ps[i] = {positions[i][0], positions[i][1], positions[i][2]};
            aabb = merge(aabb, aabb_getter<vec3>()(ps[i]));
        }

        auto &h_bvh = Engine::require<lbvh<vec3, aabb_getter<vec3> > >(entity_id);

        auto start_time = std::chrono::high_resolution_clock::now();
        h_bvh = lbvh<vec3, aabb_getter<vec3> >(ps.begin(), ps.end(), true);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> build_duration = end_time - start_time;
        Log::Info("Build LBVH in {} seconds", build_duration.count());

        auto result = query_host(h_bvh, nearest(ps[0]), distance_calculator<vec3, vec3>());
        Log::Info("Nearest to first point: {} with distance: {}", result.first, result.second);
        {
            start_time = std::chrono::high_resolution_clock::now();
            //TODO: build from host to see if there is a problem while loading
            auto &bvh = Engine::require<cuda::bvh::device_data<vec3> >(entity_id);
            bvh.objects = ps;
            bvh::construct_device(bvh);
            end_time = std::chrono::high_resolution_clock::now();

            build_duration = end_time - start_time;
            Log::Info("Build LBVH_DEVICE in {} seconds", build_duration.count());
            const auto &h_bvh_device = Engine::State().emplace_or_replace<cuda::bvh::host_data<vec3>>(entity_id, bvh::get_host_data(bvh));

            distance_calculator<vec3, vec3> dist;
            auto result = cuda::bvh::query_host<vec3>(h_bvh_device, nearest(ps[0]), dist);
            Log::Info("Nearest to first point: {} with distance: {}", result.first, result.second);
        }
    }

    [[nodiscard]] QueryResult
    KDTreeCuda::knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const {
        auto &bvh = Engine::require<lbvh<vec3, aabb_getter<vec3> > >(entity_id);
        struct distance_calculator {
            __device__ __host__
            float operator()(const vec3 &point, const vec3 &object) const noexcept {
                return (point[0] - object[0]) * (point[0] - object[0]) +
                       (point[1] - object[1]) * (point[1] - object[1]) +
                       (point[2] - object[2]) * (point[2] - object[2]);
            }
        };
        vec3 d_query = {query_point[0], query_point[1], query_point[2]};
        QueryResult result;
        result.indices.resize(1, num_closest);
        result.distances.resize(1, num_closest);
        const auto indices = query_host(bvh, knn(d_query, num_closest), distance_calculator());
        for (long i = 0; i < num_closest; ++i) {
            auto idx = indices[i];
            result.indices(0, i) = idx;
            result.distances(0, i) = length(d_query - bvh.objects_host()[idx]);
        }
        return result;
    }

    [[nodiscard]] QueryResult KDTreeCuda::radius_query(const Vector<float, 3> &query_point, float radius) const {
        /* auto &bvh = Engine::require<lbvh<vec3, aabb_getter<vec3>>>(entity_id);
         vec3 d_query = {query_point[0], query_point[1], query_point[2]};
         QueryResult result;
         std::vector<size_t> indices;
         const auto num_found = query_host(bvh, overlaps_sphere(d_query, radius), indices);
         result.indices.resize(1, num_found);
         result.distances.resize(1, num_found);
         for (long i = 0; i < num_found; ++i) {
             auto idx = indices[i];
             result.indices(0, i) = idx;
             result.distances(0, i) = length(d_query - bvh.objects_host()[idx]);
         }*/

        QueryResult result; {
            vec3 d_query = {query_point[0], query_point[1], query_point[2]};

            auto &bvh = Engine::require<cuda::bvh::device_data<vec3> >(entity_id);

            const cuda::bvh::host_data<vec3> h_bvh_device = cuda::bvh::get_host_data(bvh);

            std::vector<unsigned int> indices_host;
            cuda::bvh::query_host(h_bvh_device, overlaps_sphere(d_query, radius), indices_host);

            auto num_found = indices_host.size();
            result.indices.resize(1, num_found);
            result.distances.resize(1, num_found);
            for (long i = 0; i < num_found; ++i) {
                auto idx = indices_host[i];
                result.indices(0, i) = idx;
                result.distances(0, i) = length(d_query - bvh.objects[idx]);
            }
        }
        return result;
    }

    [[nodiscard]] QueryResult KDTreeCuda::closest_query(const Vector<float, 3> &query_point) const {
        auto &bvh = Engine::require<lbvh<vec3, aabb_getter<vec3> > >(entity_id);
        struct distance_calculator {
            __device__ __host__
            float operator()(const vec3 &point, const vec3 &object) const noexcept {
                return (point[0] - object[0]) * (point[0] - object[0]) +
                       (point[1] - object[1]) * (point[1] - object[1]) +
                       (point[2] - object[2]) * (point[2] - object[2]);
            }
        };
        vec3 d_query = {query_point[0], query_point[1], query_point[2]};
        const auto next = query_host(bvh, nearest(d_query), distance_calculator());
        QueryResult result;
        result.indices.resize(1, 1);
        result.distances.resize(1, 1);
        result.indices(0, 0) = next.first;
        result.distances(0, 0) = next.second;

        return result;
    }

    std::vector<std::uint32_t> KDTreeCuda::get_samples(unsigned int level) const {
        auto &bvh = Engine::require<lbvh<vec3, aabb_getter<vec3> > >(entity_id);
        auto samples_h = bvh.get_samples(level);
        // Copy to host
        std::vector<std::uint32_t> samples(samples_h.size());
        for (size_t i = 0; i < samples_h.size(); ++i) {
            samples[i] = samples_h[i];
        }
        return samples;
    }

    unsigned int KDTreeCuda::compute_num_levels() const {
        auto &bvh = Engine::require<lbvh<vec3, aabb_getter<vec3> > >(entity_id);
        return bvh.compute_num_levels(bvh.objects_host().size());
    }

    void KDTreeCuda::fill_samples() const {
        auto &bvh = Engine::require<lbvh<vec3, aabb_getter<vec3> > >(entity_id);
        size_t num_closest = 128;
        auto &points = bvh.objects_host();
        Eigen::Matrix<size_t, -1, -1> knns(points.size(), num_closest);
        Eigen::Matrix<float, -1, -1> dists(points.size(), num_closest);
        for (size_t i = 0; i < points.size(); ++i) {
            auto result = knn_query({points[i][0], points[i][1], points[i][2]}, num_closest);

            auto indices = std::vector<size_t>(result.indices.data(), result.indices.data() + result.indices.size());
            auto distances = std::vector<float>(result.distances.data(),
                                                result.distances.data() + result.distances.size());
            //sort by distance in ascending order
            std::vector<size_t> sorted_indices(num_closest);
            std::iota(sorted_indices.begin(), sorted_indices.end(), 0); // Create index mapping.

            std::sort(sorted_indices.begin(), sorted_indices.end(), [&distances](size_t a, size_t b) {
                return distances[a] < distances[b];
            });

            std::vector<size_t> sorted_knns;
            std::vector<float> sorted_dists;
            for (long j = 0; j < num_closest; ++j) {
                knns(i, j) = indices[sorted_indices[j]];
                dists(i, j) = distances[sorted_indices[j]];
            }
        }
        //TODO fill samples
        //the sampling i get also depends on how deep i descend in the tree... I need to think about what i want here...

        //bvh.fill_samples_new(knns.block(0, 1, points.size(), num_closest -1), dists.block(0, 1, points.size(), num_closest -1)); //somehow broken
        //bvh.fill_samples(knns.block(0, 1, points.size(), num_closest -1), dists.block(0, 1, points.size(), num_closest -1)); //better than fill_samples_new but still not really blue noise
        bvh.fill_samples_closest_to_center(knns.block(0, 1, points.size(), num_closest - 1),
                                           dists.block(0, 1, points.size(), num_closest - 1));
    }
}
