//
// Created by alex on 6/15/25.
//

#include "GLFWContext.h"
#include "Logger.h"
#include <GLFW/glfw3.h>

namespace Bcg {
    static bool initialized = false;

    void GLFWContext::glfw_error_callback(int error, const char *description) {
        Log::Error("GLFW Error {}, {}", error, description);
    }

    bool GLFWContext::init() {
        if (initialized) {
            return true;
        }
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit()) {
            Log::Error("Failed to initialize GLFW context");
            return false;
        }
        Log::Info("Initialized GLFW context");
        initialized = true;
        return true;
    }

    void GLFWContext::shutdown() {
        if (!initialized) {
            return;
        }

        glfwTerminate();
        Log::Info("GLFW context destroyed");
        initialized = false;
    }
}
