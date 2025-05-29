#ifndef LBVH_MORTON_CODE_CUH
#define LBVH_MORTON_CODE_CUH

#include "mat_vec.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace Bcg::cuda {

    __device__ __host__
    inline std::uint32_t expand_bits(std::uint32_t v) noexcept {
        v = (v * 0x00010001u) & 0xFF0000FFu;
        v = (v * 0x00000101u) & 0x0F00F00Fu;
        v = (v * 0x00000011u) & 0xC30C30C3u;
        v = (v * 0x00000005u) & 0x49249249u;
        return v;
    }

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
    __device__ __host__
    inline std::uint32_t morton_code(vec3 xyz, float resolution = 1024.0f) noexcept {
        xyz[0] = ::fminf(::fmaxf(xyz[0] * resolution, 0.0f), resolution - 1.0f);
        xyz[1] = ::fminf(::fmaxf(xyz[1] * resolution, 0.0f), resolution - 1.0f);
        xyz[2] = ::fminf(::fmaxf(xyz[2] * resolution, 0.0f), resolution - 1.0f);
        const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz[0]));
        const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz[1]));
        const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz[2]));
        return xx * 4 + yy * 2 + zz;
    }

    __device__
    inline int common_upper_bits(const unsigned int lhs, const unsigned int rhs) noexcept {
        return ::__clz(lhs ^ rhs);
    }

    __device__
    inline int common_upper_bits(const unsigned long long int lhs, const unsigned long long int rhs) noexcept {
        return ::__clzll(lhs ^ rhs);
    }

} // lbvh
#endif// LBVH_MORTON_CODE_CUH
