//
// Created by alex on 08.08.24.
//

#include "KDTreeCuda.h"
#include "lbvh/lbvh.cuh"
#include "lbvh/query.cuh"
#include "Engine.h"

#include <vector>

namespace Bcg {
    struct aabb_getter {
        __device__
        lbvh::aabb<float> operator()(const float4 f) const noexcept {
            lbvh::aabb<float> retval;
            retval.upper = f;
            retval.lower = f;
            return retval;
        }
    };

    struct sphere_getter {
        __device__ __host__
        lbvh::sphere<float> operator()(const float4 center, float radius) const noexcept {
            lbvh::sphere<float> retval;
            retval.center = center;
            retval.radius = radius;
            return retval;
        }
    };

    KDTreeCuda::KDTreeCuda(entt::entity entity_id) : entity_id(entity_id) {

    }

    KDTreeCuda::~KDTreeCuda() = default;

    KDTreeCuda::operator bool() const {
        return Engine::has<lbvh::bvh<float, float4, aabb_getter>>(entity_id);
    }

    void KDTreeCuda::build(const std::vector<Vector<float, 3>> &positions) {
        std::vector<float4> ps(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            ps[i] = {positions[i].x(), positions[i].y(), positions[i].z()};
        }
        auto &bvh = Engine::require<lbvh::bvh<float, float4, aabb_getter>>(entity_id);
        bvh = lbvh::bvh<float, float4, aabb_getter>(ps.begin(), ps.end(), true);
    }

    [[nodiscard]] QueryResult
    KDTreeCuda::knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const {
        //TODO this is still wrong
        auto &bvh = Engine::require<lbvh::bvh<float, float4, aabb_getter>>(entity_id);
        struct distance_calculator {
            __device__ __host__
            float operator()(const float4 point, const float4 object) const noexcept {
                return (point.x - object.x) * (point.x - object.x) +
                       (point.y - object.y) * (point.y - object.y) +
                       (point.z - object.z) * (point.z - object.z);
            }
        };
        const auto bvh_dev = bvh.get_device_repr();
        float3 d_query = {query_point[0], query_point[1], query_point[2]};
        QueryResult result;
        const auto indices = lbvh::query_host(bvh, lbvh::knn(d_query, num_closest), distance_calculator());
        for (const auto idx: indices) {
            result.indices.push_back(idx);
        }
        return result;
    }

    [[nodiscard]] QueryResult KDTreeCuda::radius_query(const Vector<float, 3> &query_point, float radius) const {

        return {};
    }

    [[nodiscard]] QueryResult KDTreeCuda::closest_query(const Vector<float, 3> &query_point) const {
        auto &bvh = Engine::require<lbvh::bvh<float, float4, aabb_getter>>(entity_id);
        struct distance_calculator {
            __device__ __host__
            float operator()(const float4 point, const float4 object) const noexcept {
                return (point.x - object.x) * (point.x - object.x) +
                       (point.y - object.y) * (point.y - object.y) +
                       (point.z - object.z) * (point.z - object.z);
            }
        };
        const auto bvh_dev = bvh.get_device_repr();
        float3 d_query = {query_point[0], query_point[1], query_point[2]};
        const auto nest = lbvh::query_host(bvh, lbvh::nearest(d_query),
                                           distance_calculator());
        QueryResult result;
        result.indices.emplace_back(nest.first);
        result.distances.emplace_back(nest.second);
        return result;
        /*       return {};*/
    }

}