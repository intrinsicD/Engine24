//
// Created by alex on 30.07.24.
//

#ifndef ENGINE24_CUDACOMMON_H
#define ENGINE24_CUDACOMMON_H

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <math_constants.h>
#include <vector_types.h>
#include <limits>

#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_HOST_DEVICE CUDA_HOST CUDA_DEVICE
#else
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_HOST_DEVICE
#endif

namespace Bcg {
    template<typename T>
    CUDA_HOST_DEVICE
    inline T infinity() noexcept;

/*
    template<>
    CUDA_HOST_DEVICE
    inline float infinity<float>() noexcept { return CUDART_INF; }

    template<>
    CUDA_HOST_DEVICE
    inline double infinity<double>() noexcept { return CUDART_INF; }
*/

    template<>
    CUDA_HOST_DEVICE
    inline float infinity<float>() noexcept { return std::numeric_limits<float>::infinity(); }

    template<>
    CUDA_HOST_DEVICE
    inline double infinity<double>() noexcept { return std::numeric_limits<double>::infinity(); }
}

#endif //ENGINE24_CUDACOMMON_H
