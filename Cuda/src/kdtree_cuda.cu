//
// Created by alex on 08.08.24.
//

#include "Cuda/KDTreeCuda.h"
#include "lbvh.cuh"
#include "bvh_device.cuh"
#include "Engine.h"

#include <vector>
#include <numeric>

namespace Bcg::cuda { ;

    KDTreeCuda::KDTreeCuda(entt::entity entity_id) : entity_id(entity_id) {

    }

    KDTreeCuda::~KDTreeCuda() = default;

    KDTreeCuda::operator bool() const {
        return Engine::has<lbvh<glm::vec3, aabb_getter<glm::vec3>>>(entity_id);
    }

    void KDTreeCuda::build(const std::vector<Vector<float, 3>> &positions) {
        std::vector<glm::vec3> ps(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            ps[i] = {positions[i].x, positions[i].y, positions[i].z};
        }

        auto &h_bvh = Engine::require<lbvh<glm::vec3, aabb_getter<glm::vec3>>>(entity_id);
        h_bvh = lbvh<glm::vec3, aabb_getter<glm::vec3>>(ps.begin(), ps.end(), true);
    }

    [[nodiscard]] QueryResult
    KDTreeCuda::knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const {
        auto &bvh = Engine::require<lbvh<glm::vec3, aabb_getter<glm::vec3>>>(entity_id);
        struct distance_calculator {
            __device__ __host__
            float operator()(const glm::vec3 &point, const glm::vec3 &object) const noexcept {
                return (point.x - object.x) * (point.x - object.x) +
                       (point.y - object.y) * (point.y - object.y) +
                       (point.z - object.z) * (point.z - object.z);
            }
        };
        glm::vec3 d_query = {query_point[0], query_point[1], query_point[2]};
        QueryResult result;
        const auto indices = query_host(bvh, knn(d_query, num_closest), distance_calculator());
        for (const auto idx: indices) {
            result.indices.push_back(idx);
        }
        return result;
    }

    [[nodiscard]] QueryResult KDTreeCuda::radius_query(const Vector<float, 3> &query_point, float radius) const {
        auto &bvh = Engine::require<lbvh<glm::vec3, aabb_getter<glm::vec3>>>(entity_id);
        glm::vec3 d_query = {query_point[0], query_point[1], query_point[2]};
        QueryResult result;
        std::vector<size_t> indices;
        const auto num_found = query_host(bvh, overlaps_sphere(d_query, radius), indices);
        result.indices.reserve(num_found);
        for (const auto idx: indices) {
            result.indices.push_back(idx);
        }
        return result;
    }

    [[nodiscard]] QueryResult KDTreeCuda::closest_query(const Vector<float, 3> &query_point) const {
        auto &bvh = Engine::require<lbvh<glm::vec3, aabb_getter<glm::vec3>>>(entity_id);
        struct distance_calculator {
            __device__ __host__
            float operator()(const glm::vec3 &point, const glm::vec3 &object) const noexcept {
                return (point.x - object.x) * (point.x - object.x) +
                       (point.y - object.y) * (point.y - object.y) +
                       (point.z - object.z) * (point.z - object.z);
            }
        };
        glm::vec3 d_query = {query_point[0], query_point[1], query_point[2]};
        const auto next = query_host(bvh, nearest(d_query), distance_calculator());
        QueryResult result;
        result.indices.emplace_back(next.first);
        result.distances.emplace_back(next.second);
        return result;
    }

    std::vector<std::uint32_t> KDTreeCuda::get_samples(unsigned int level) const {
        auto &bvh = Engine::require<lbvh<glm::vec3, aabb_getter<glm::vec3>>>(entity_id);
        auto samples_h = bvh.get_samples(level);
        // Copy to host
        std::vector<std::uint32_t> samples(samples_h.size());
        for (size_t i = 0; i < samples_h.size(); ++i) {
            samples[i] = samples_h[i];
        }
        return samples;
    }

    unsigned int KDTreeCuda::compute_num_levels() const{
        auto &bvh = Engine::require<lbvh<glm::vec3, aabb_getter<glm::vec3>>>(entity_id);
        return bvh.compute_num_levels(bvh.objects_host().size());
    }

    void KDTreeCuda::fill_samples() const {
        auto &bvh = Engine::require<lbvh<glm::vec3, aabb_getter<glm::vec3>>>(entity_id);
        size_t num_closest = 64;
        auto &points = bvh.objects_host();
        std::vector<std::vector<size_t>> knns(points.size());
        std::vector<std::vector<float>> dists(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            std::vector<size_t> indices = knn_query({points[i].x, points[i].y, points[i].z}, num_closest).indices;
            std::vector<float> distances;

            for(size_t j = 0; j < knns[i].size(); ++j){
                distances.push_back((points[i].x - points[knns[i][j]].x) * (points[i].x - points[knns[i][j]].x) +
                                   (points[i].y - points[knns[i][j]].y) * (points[i].y - points[knns[i][j]].y) +
                                   (points[i].z - points[knns[i][j]].z) * (points[i].z - points[knns[i][j]].z));
            }
            //sort by distance in ascending order
            std::vector<size_t> sorted_indices(knns[i].size());
            std::iota(sorted_indices.begin(), sorted_indices.end(), 0); // Create index mapping.

            std::sort(sorted_indices.begin(), sorted_indices.end(), [&distances](size_t a, size_t b) {
                return distances[a] < distances[b];
            });

            std::vector<size_t> sorted_knns;
            std::vector<float> sorted_dists;
            for (size_t idx : sorted_indices) {
                sorted_knns.push_back(indices[idx]);
                sorted_dists.push_back(distances[idx]);
            }

            knns[i] = std::move(sorted_knns);
            dists[i] = std::move(sorted_dists);
        }
        bvh.fill_samples(&knns, &dists);
    }

}