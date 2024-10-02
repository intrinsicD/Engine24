//
// Created by alex on 08.08.24.
//

#include <cuda_runtime_api.h>
#include "CudaCommon.cuh"

namespace Bcg::cuda {
    __global__ void kernel() {
        printf("Hello from CUDA Device!\n");
    }

    CudaError HelloFromCudaDevice() {
        kernel<<<1, 1>>>();
        return CudaCheckErrorAndSync();
    }

    CudaError CudaCheckErrorAndSync() {
        // Check for any errors launching the kernel
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return CudaError::FailedKernelLaunch;
        }

        // Synchronize device
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            //Log::Error("Failed to synchronize: " + func_name + "  " + std::string(cudaGetErrorString(err)));
            return CudaError::FailedSynchronisation;
        }
        return CudaError::None;
    }
}