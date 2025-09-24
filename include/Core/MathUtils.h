#pragma once

#include "Macros.h"

namespace Bcg {
    template<typename T>
    CUDA_HOST_DEVICE T d_clamp(T v, T lo, T hi) noexcept {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    template<typename T>
    CUDA_HOST_DEVICE T d_abs(T v) noexcept { return v < T(0) ? -v : v; }

    template<typename T>
    CUDA_HOST_DEVICE T d_max(T a, T b) noexcept { return a < b ? b : a; }

    template<typename T>
    CUDA_HOST_DEVICE T d_min(T a, T b) noexcept { return a < b ? a : b; }

    template<class T>
    CUDA_HOST_DEVICE T d_sqrt(T x) {
#ifdef __CUDA_ARCH__
        return sqrt(x); // device overload
#else
        using std::sqrt;
        return sqrt(x);
#endif
    }
}
