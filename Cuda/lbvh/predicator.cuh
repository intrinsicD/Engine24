#ifndef LBVH_PREDICATOR_CUH
#define LBVH_PREDICATOR_CUH

#include "aabb.cuh"
#include "sphere.cuh"

namespace lbvh {

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
    };

    __device__ __host__
    inline query_overlap<aabb<float>> overlaps_aabb(const aabb<float> &region) noexcept {
        return query_overlap<aabb<float>>(region);
    }

    __device__ __host__
    inline query_overlap<sphere<float>> overlaps_sphere(const sphere<float> &region) noexcept {
        return query_overlap<sphere<float>>(region);
    }

    __device__ __host__
    inline query_overlap<sphere<float>> overlaps_sphere(const float3 &point, float radius) noexcept {
        return query_overlap<sphere<float>>({make_float4(point.x, point.y, point.z, radius)});
    }

    template<typename Real>
    struct query_nearest {
        // float4/double4
        using vector_type = typename vector_of<Real>::type;

        __device__ __host__
        query_nearest(const vector_type &tgt) : target(tgt) {}

        query_nearest() = default;

        ~query_nearest() = default;

        query_nearest(const query_nearest &) = default;

        query_nearest(query_nearest &&) = default;

        query_nearest &operator=(const query_nearest &) = default;

        query_nearest &operator=(query_nearest &&) = default;

        vector_type target;
    };

    __device__ __host__
    inline query_nearest<float> nearest(const float4 &point) noexcept {
        return query_nearest<float>(point);
    }

    __device__ __host__
    inline query_nearest<float> nearest(const float3 &point) noexcept {
        return query_nearest<float>(make_float4(point.x, point.y, point.z, 0.0f));
    }

    __device__ __host__
    inline query_nearest<double> nearest(const double4 &point) noexcept {
        return query_nearest<double>(point);
    }

    __device__ __host__
    inline query_nearest<double> nearest(const double3 &point) noexcept {
        return query_nearest<double>(make_double4(point.x, point.y, point.z, 0.0));
    }

    template<typename Real>
    struct query_knn {
        // float4/double4
        using vector_type = typename vector_of<Real>::type;

        __device__ __host__
        query_knn(const vector_type &tgt, unsigned int k_closest) : target(tgt), k_closest(k_closest) {}

        query_knn() = default;

        ~query_knn() = default;

        query_knn(const query_knn &) = default;

        query_knn(query_knn &&) = default;

        query_knn &operator=(const query_knn &) = default;

        query_knn &operator=(query_knn &&) = default;

        vector_type target;
        unsigned int k_closest;
    };

    __device__ __host__
    inline query_knn<float> knn(const float4 &point, unsigned int k) noexcept {
        return query_knn<float>(point, k);
    }

    __device__ __host__
    inline query_knn<float> knn(const float3 &point, unsigned int k) noexcept {
        return query_knn<float>(make_float4(point.x, point.y, point.z, 0.0f), k);
    }

    __device__ __host__
    inline query_knn<double> knn(const double4 &point, unsigned int k) noexcept {
        return query_knn<double>(point, k);
    }

    __device__ __host__
    inline query_knn<double> knn(const double3 &point, unsigned int k) noexcept {
        return query_knn<double>(make_double4(point.x, point.y, point.z, 0.0), k);
    }

} // lbvh
#endif// LBVH_PREDICATOR_CUH
