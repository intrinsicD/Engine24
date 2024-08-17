//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_GAUSSIAN_CUH
#define ENGINE24_GAUSSIAN_CUH

#include "vec3.cuh"

namespace Bcg::cuda{
    struct gaussian {
        vec3 mean;
        float cov[16];
    };
}

#endif //ENGINE24_GAUSSIAN_CUH
