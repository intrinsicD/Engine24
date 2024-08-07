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
#define CUDA_KERNEL __global__
#else
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_HOST_DEVICE
#define CUDA_KERNEL
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

    template<typename T>
    struct vec3 {
        T x, y, z;

        CUDA_HOST_DEVICE
        vec3() : x(0.0f), y(0.0f), z(0.0f) {}

        CUDA_HOST_DEVICE
        vec3(float x, float y, float z) : x(x), y(y), z(z) {}

        CUDA_HOST_DEVICE
        T &operator[](int i) {
            return ((&x)[i]);
        }

        CUDA_HOST_DEVICE
        const T &operator[](int i) const {
            return ((&x)[i]);
        }
    };

    template<typename T>
    struct CapturedBuffer {
        T *data = nullptr;
        size_t bytes = 0;

        cudaGraphicsResource *cuda_resource = nullptr;

        CapturedBuffer() = default;

        explicit CapturedBuffer(unsigned int opengl_buffer) {
            cudaGraphicsGLRegisterBuffer(&cuda_resource, opengl_buffer, cudaGraphicsMapFlagsNone);
            cudaGraphicsMapResources(1, &cuda_resource, nullptr);
            cudaGraphicsResourceGetMappedPointer((void **) &data, &bytes, cuda_resource);
        }

        ~CapturedBuffer() {
            cudaGraphicsUnmapResources(1, &cuda_resource, nullptr);
            cudaGraphicsUnregisterResource(cuda_resource);
            data = nullptr;
            bytes = 0;
        }

        operator bool() const {
            return data != nullptr && bytes != 0;
        }
    };
}

#endif //ENGINE24_CUDACOMMON_H
