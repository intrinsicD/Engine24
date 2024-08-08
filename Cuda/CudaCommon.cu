//
// Created by alex on 08.08.24.
//

#include <cuda_runtime_api.h>

#include "CudaCommon.cuh"
#include "Logger.h"

namespace Bcg {
    __global__ void kernel() {
        printf("Hello from CUDA Device!\n");
    }

    void HelloFromCudaDevice() {
        kernel<<<1, 1>>>();
        if (CudaCheckErrorAndSync(__func__)) {
            Log::Info("CUDA kernel " + std::string(__func__) + " executed successfully.");
        }
    }

    bool CudaCheckErrorAndSync(const std::string &func_name) {
        // Check for any errors launching the kernel
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            Log::Error("Failed to launch kernel: " + func_name + " " + std::string(cudaGetErrorString(err)));
            return false;
        }

        // Synchronize device
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            Log::Error("Failed to synchronize: " + func_name + "  " + std::string(cudaGetErrorString(err)));
            return false;
        }
        return true;
    }
}