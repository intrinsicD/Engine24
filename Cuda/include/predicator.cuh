#ifndef LBVH_PREDICATOR_CUH
#define LBVH_PREDICATOR_CUH

#include "aabb.cuh"
#include "sphere_utils.cuh"

namespace Bcg::cuda {
    template<typename Object>
    struct query_overlap {
        __device__ __host__
        query_overlap(const Object &tgt) : target(tgt) {}

        query_overlap() = default;

        ~query_overlap() = default;

        query_overlap(const query_overlap &) = default;

        query_overlap(query_overlap &&) = default;

        query_overlap &operator=(const query_overlap &) = default;

        query_overlap &operator=(query_overlap &&) = default;

        __device__ __host__
        inline bool operator()(const Object &object) noexcept {
            return intersects(object, target);
        }

        Object target;
        unsigned int num_found = 0;
    };

    template<typename Object>
    struct query_overlap_count {
        __device__ __host__
        query_overlap_count(const Object &tgt) : target(tgt) {}

        query_overlap_count() = default;

        ~query_overlap_count() = default;

        query_overlap_count(const query_overlap_count &) = default;

        query_overlap_count(query_overlap_count &&) = default;

        query_overlap_count &operator=(const query_overlap_count &) = default;

        query_overlap_count &operator=(query_overlap_count &&) = default;

        __device__ __host__
        inline bool operator()(const Object &object) noexcept {
            return intersects(object, target);
        }

        Object target;
    };

    __device__ __host__
    inline query_overlap<aabb> overlaps_aabb(const aabb &region) noexcept {
        return query_overlap<aabb>(region);
    }

    __device__ __host__
    inline query_overlap<sphere> overlaps_sphere(const sphere &region) noexcept {
        return query_overlap<sphere>(region);
    }

    __device__ __host__
    inline query_overlap<sphere> overlaps_sphere(const vec3 &point, float radius) noexcept {
        return query_overlap<sphere>({point, radius});
    }

    struct query_nearest {
        __device__ __host__
        query_nearest(const vec3 &tgt) : target(tgt) {}

        query_nearest() = default;

        ~query_nearest() = default;

        query_nearest(const query_nearest &) = default;

        query_nearest(query_nearest &&) = default;

        query_nearest &operator=(const query_nearest &) = default;

        query_nearest &operator=(query_nearest &&) = default;

        vec3 target;
    };

    __device__ __host__
    inline query_nearest nearest(const vec3 &point) noexcept {
        return query_nearest(point);
    }

    struct query_knn {
        __device__ __host__
        query_knn(const vec3 &tgt, unsigned int k_closest) : target(tgt), k_closest(k_closest) {}

        query_knn() = default;

        ~query_knn() = default;

        query_knn(const query_knn &) = default;

        query_knn(query_knn &&) = default;

        query_knn &operator=(const query_knn &) = default;

        query_knn &operator=(query_knn &&) = default;

        vec3 target;
        unsigned int k_closest;
    };

    __device__ __host__
    inline query_knn knn(const vec3 &point, unsigned int k) noexcept {
        return query_knn(point, k);
    }

    template<typename ObjectType, typename QueryObjectType>
    struct distance_calculator;

    template<>
    struct distance_calculator<vec3, vec3> {
        __device__ __host__
        float operator()(const vec3 &point, const vec3 &object) const noexcept {
            return (point[0] - object[0]) * (point[0] - object[0]) +
                   (point[1] - object[1]) * (point[1] - object[1]) +
                   (point[2] - object[2]) * (point[2] - object[2]);
        }
    };
} // lbvh
#endif// LBVH_PREDICATOR_CUH
