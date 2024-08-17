#ifndef LBVH_UTILITY_CUH
#define LBVH_UTILITY_CUH

#include <vector_types.h>
#include <math_constants.h>
#include <cuda_runtime.h>

namespace Bcg::cuda {
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
