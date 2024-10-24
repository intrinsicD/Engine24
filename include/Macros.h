//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_MACROS_H
#define ENGINE24_MACROS_H

namespace Bcg {
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CUDA_HOST
#endif
}

#endif //ENGINE24_MACROS_H
