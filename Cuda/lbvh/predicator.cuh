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

    template<typename Real>
    __device__ __host__
    query_overlap<aabb<Real>> overlaps(const aabb<Real> &region) noexcept {
        return query_overlap<aabb<Real>>(region);
    }

    template<typename Real>
    __device__ __host__
    query_overlap<sphere<Real>> overlaps(const sphere<Real> &region) noexcept {
        return query_overlap<sphere<Real>>(region);
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
    struct query_k_closest {
        // float4/double4
        using vector_type = typename vector_of<Real>::type;

        __device__ __host__
        query_k_closest(const vector_type &tgt, unsigned int k_closest) : target(tgt), k_closest(k_closest) {}

        query_k_closest() = default;

        ~query_k_closest() = default;

        query_k_closest(const query_k_closest &) = default;

        query_k_closest(query_k_closest &&) = default;

        query_k_closest &operator=(const query_k_closest &) = default;

        query_k_closest &operator=(query_k_closest &&) = default;

        vector_type target;
        unsigned int k_closest;
    };

    __device__ __host__
    inline query_k_closest<float> knn(const float4 &point, unsigned int k) noexcept {
        return query_k_closest<float>(point, k);
    }

    __device__ __host__
    inline query_k_closest<float> knn(const float3 &point, unsigned int k) noexcept {
        return query_k_closest<float>(make_float4(point.x, point.y, point.z, 0.0f), k);
    }

    __device__ __host__
    inline query_k_closest<double> knn(const double4 &point, unsigned int k) noexcept {
        return query_k_closest<double>(point, k);
    }

    __device__ __host__
    inline query_k_closest<double> knn(const double3 &point, unsigned int k) noexcept {
        return query_k_closest<double>(make_double4(point.x, point.y, point.z, 0.0), k);
    }

} // lbvh
#endif// LBVH_PREDICATOR_CUH
