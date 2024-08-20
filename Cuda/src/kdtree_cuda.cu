//
// Created by alex on 08.08.24.
//

#include "Cuda/KDTreeCuda.h"
#include "lbvh.cuh"
#include "Engine.h"
#include "Logger.h"

#include <vector>

namespace Bcg::cuda {


    using aabb_getter = aabb_getter;

    KDTreeCuda::KDTreeCuda(entt::entity entity_id) : entity_id(entity_id) {

    }

    KDTreeCuda::~KDTreeCuda() = default;

    KDTreeCuda::operator bool() const {
        return Engine::has<lbvh<vec3, aabb_getter>>(entity_id);
    }

    void KDTreeCuda::build(const std::vector<Vector<float, 3>> &positions) {
        std::vector<vec3> ps(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            ps[i] = {positions[i].x(), positions[i].y(), positions[i].z()};
        }
        auto &bvh = Engine::require<lbvh<vec3, aabb_getter>>(entity_id);
        bvh = lbvh<vec3, aabb_getter>(ps.begin(), ps.end(), true);
    }

    [[nodiscard]] QueryResult
    KDTreeCuda::knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const {
        auto &bvh = Engine::require<lbvh<vec3, aabb_getter>>(entity_id);
        struct distance_calculator {
            __device__ __host__
            float operator()(const vec3 &point, const vec3 &object) const noexcept {
                return (point.x - object.x) * (point.x - object.x) +
                       (point.y - object.y) * (point.y - object.y) +
                       (point.z - object.z) * (point.z - object.z);
            }
        };
        vec3 d_query = {query_point[0], query_point[1], query_point[2]};
        QueryResult result;
        const auto indices = query_host(bvh, knn(d_query, num_closest), distance_calculator());
        for (const auto idx: indices) {
            result.indices.push_back(idx);
        }
        return result;
    }

    [[nodiscard]] QueryResult KDTreeCuda::radius_query(const Vector<float, 3> &query_point, float radius) const {
        auto &bvh = Engine::require<lbvh<vec3, aabb_getter>>(entity_id);
        vec3 d_query = {query_point[0], query_point[1], query_point[2]};
        QueryResult result;
        std::vector<size_t> indices;
        Log::Info("query_point {} {} {}", query_point[0], query_point[1], query_point[2]);
        const auto num_found = query_host(bvh, overlaps_sphere(d_query, radius), indices);
        result.indices.reserve(num_found);
        for (const auto idx: indices) {
            result.indices.push_back(idx);
        }
        return result;
    }

    [[nodiscard]] QueryResult KDTreeCuda::closest_query(const Vector<float, 3> &query_point) const {
        auto &bvh = Engine::require<lbvh<vec3, aabb_getter>>(entity_id);
        struct distance_calculator {
            __device__ __host__
            float operator()(const vec3 &point, const vec3 &object) const noexcept {
                return (point.x - object.x) * (point.x - object.x) +
                       (point.y - object.y) * (point.y - object.y) +
                       (point.z - object.z) * (point.z - object.z);
            }
        };
        vec3 d_query = {query_point[0], query_point[1], query_point[2]};
        const auto next = query_host(bvh, nearest(d_query), distance_calculator());
        QueryResult result;
        result.indices.emplace_back(next.first);
        result.distances.emplace_back(next.second);
        return result;
    }

}