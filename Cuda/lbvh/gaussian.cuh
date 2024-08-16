//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_GAUSSIAN_CUH
#define ENGINE24_GAUSSIAN_CUH

#include "utility.cuh"

namespace lbvh{
    template<typename T>
    struct gaussian {
        typename vector_of<T>::type mean;
        float cov[16];
    };
}

#endif //ENGINE24_GAUSSIAN_CUH
