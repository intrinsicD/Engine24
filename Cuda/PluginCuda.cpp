//
// Created by alex on 30.07.24.
//

#include "PluginCuda.h"
#include "Logger.h"
#include "CudaCommon.h"

namespace Bcg {
    void PluginCuda::init() {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            Log::Error("No CUDA-capable devices found");
        }
        cudaSetDevice(0); // Choose the first device for simplicity
    }

    void PluginCuda::activate() {
        Plugin::activate();
    }

    void PluginCuda::begin_frame() {}

    void PluginCuda::update() {}

    void PluginCuda::end_frame() {}

    void PluginCuda::deactivate() {
        Plugin::deactivate();
    }

    void PluginCuda::render_menu() {}

    void PluginCuda::render_gui() {}

    void PluginCuda::render() {}
}