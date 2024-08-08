#ifndef LBVH_UTILITY_CUH
#define LBVH_UTILITY_CUH

#include <vector_types.h>
#include <math_constants.h>
#include <cuda_runtime.h>

namespace lbvh {

    template<typename T>
    struct vector_of;
    template<>
    struct vector_of<float> {
        using type = float4;
    };
    template<>
    struct vector_of<double> {
        using type = double4;
    };

    template<typename T>
    using vector_of_t = typename vector_of<T>::type;

    template<typename T>
    __device__ __host__
    inline T infinity() noexcept;

    template<>
    __device__ __host__
    inline float infinity<float>() noexcept {
#ifdef __CUDA_ARCH__
        return CUDART_INF_F;
#else
        return std::numeric_limits<float>::infinity();
#endif
    }

    template<>
    __device__ __host__
    inline double infinity<double>() noexcept {
#ifdef __CUDA_ARCH__
        return CUDART_INF;
#else
        return std::numeric_limits<float>::infinity();
#endif
    }

} // lbvh
#endif// LBVH_UTILITY_CUH
