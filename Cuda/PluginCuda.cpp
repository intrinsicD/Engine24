//
// Created by alex on 30.07.24.
//

#include "PluginCuda.h"
#include "CudaTest.cuh"


namespace Bcg {
    void PluginCuda::activate() {
        Plugin::activate();

        HelloFromCudaDevice();
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