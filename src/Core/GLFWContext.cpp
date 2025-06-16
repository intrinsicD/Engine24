//
// Created by alex on 6/15/25.
//

#include "Platform.h"
#include "Logger.h"
#include <GLFW/glfw3.h>

namespace Bcg {
    static bool initialized = false;

    void Platform::glfw_error_callback(int error, const char *description) {
        Log::Error("GLFW Error {}, {}", error, description);
    }

    Platform::Platform(){
        if (initialized) {
            return;
        }

        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit()) {
            Log::Error("Failed to initialize GLFW context");
            return;
        }
        Log::Info("Initialized GLFW context");
        initialized = true;
    }

    Platform::~Platform() {
        if (!initialized) {
            return;
        }

        glfwTerminate();
        Log::Info("GLFW context destroyed");
        initialized = false;
    }

    bool Platform::is_initialized() {
        return initialized;
    }
}
