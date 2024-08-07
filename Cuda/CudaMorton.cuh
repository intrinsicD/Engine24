//
// Created by alex on 07.08.24.
//

#ifndef ENGINE24_CUDAMORTON_CUH
#define ENGINE24_CUDAMORTON_CUH

#include "CudaCommon.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

namespace Bcg::Cuda {
    CUDA_DEVICE
    inline unsigned long InterleaveBits(unsigned int x, unsigned int y, unsigned int z) {
        unsigned long morton_code = 0;
        for (int i = 0; i < (sizeof(unsigned int) * 8); ++i) {
            morton_code |= ((x & ((unsigned long) 1 << i)) << (2 * i)) |
                           ((y & ((unsigned long) 1 << i)) << (2 * i + 1)) |
                           ((z & ((unsigned long) 1 << i)) << (2 * i + 2));
        }
        return morton_code;
    }

    CUDA_DEVICE
    inline unsigned long ComputeMortonCodeDevice(const float3 &point, float min_value, float max_value) {
        unsigned int x = static_cast<unsigned int >((point.x - min_value) / (max_value - min_value) * 1023.0f);
        unsigned int y = static_cast<unsigned int >((point.y - min_value) / (max_value - min_value) * 1023.0f);
        unsigned int z = static_cast<unsigned int >((point.z - min_value) / (max_value - min_value) * 1023.0f);
        return InterleaveBits(x, y, z);
    }

    CUDA_KERNEL void
    ComputeMortonCodesKernel(const float3 *points, unsigned long *morton_codes, int num_points, float min_value,
                             float max_value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_points) {
            morton_codes[idx] = ComputeMortonCodeDevice(points[idx], min_value, max_value);
        }
    }

    CUDA_DEVICE
    inline void SortMortonCodesDevice(float3 *points, unsigned int num_points, float min_value, float max_value) {
        thrust::sort(thrust::device, points, points + num_points,
                     CUDA_DEVICE [min_value, max_value](const float3 &a, const float3 &b) {
                    return ComputeMortonCode(a, min_value, max_value) < ComputeMortonCode(b, min_value, max_value);
                });
    }

    CUDA_HOST
    inline void SortMortonCodesHost(const float3 *points, unsigned int num_points, float min_value, float max_value) {
        float3 *d_points;
        cudaMalloc(&d_points, num_points * sizeof(float3));
        cudaMemcpy(d_points, points, num_points * sizeof(float3), cudaMemcpyHostToDevice);

        unsigned long *d_morton_codes;
        cudaMalloc(&d_morton_codes, num_points * sizeof(unsigned long));

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
        ComputeMortonCodesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_morton_codes, num_points, min_value,
                                                                     max_value);
        cudaDeviceSynchronize();

        thrust::device_ptr<unsigned long> thrust_d_morton_codes = thrust::device_pointer_cast(d_morton_codes);
        thrust::device_ptr <float3> thrust_d_points = thrust::device_pointer_cast(d_points);
        thrust::sort_by_key(thrust_d_morton_codes, thrust_d_morton_codes + num_points, thrust_d_points);

        cudaMemcpy(h_points.data(), d_points, num_points * sizeof(float3), cudaMemcpyDeviceToHost);

        for (int i = 0; i < num_points; ++i) {
            points[i] = {h_points[i].x, h_points[i].y, h_points[i].z};
        }

        cudaFree(d_points);
        cudaFree(d_morton_codes);
    }
}

#endif //ENGINE24_CUDAMORTON_CUH
