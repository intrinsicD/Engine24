//
// Created by alex on 23.07.24.
//

#if defined(_WIN32) || defined(_WIN64)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#endif

#include "Application.h"
#include "Logger.h"
#include "Color.h"
#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"
#include "bgfx/platform.h"

namespace Bcg {
    static GLFWwindow *window;

    Application::Application() {

    }

    void Application::init(int width, int height, const char *title) {
        if (!glfwInit()) {
            Log::Error("Could not initialize glfw!");
        }

        // Set GLFW to not create an OpenGL context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // Create GLFW window
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            Bcg::Log::Error("Could not create glfw window!");
        }

        glfwMakeContextCurrent(window);

        bgfx.type = bgfx::RendererType::Count; // Automatically choose renderer
        bgfx.resolution.width = width;
        bgfx.resolution.height = height;
        bgfx.resolution.reset = BGFX_RESET_VSYNC;

#if defined(_WIN32) || defined(_WIN64)
        m_bgfxInit.platformData.nwh = glfwGetWin32Window(m_window);
#elif defined(__APPLE__)
        m_bgfxInit.platformData.nwh = glfwGetCocoaWindow(m_window);
#elif defined(__linux__)
        bgfx.platformData.ndt = glfwGetX11Display();
        bgfx.platformData.nwh = (void *) (uintptr_t) glfwGetX11Window(window);
#endif

        if (!bgfx::init(bgfx)) {
            Log::Error("Could not initialize BGFX!");
        }

        bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, floatColorToUint32(0.2, 0.3, 0.3, 1.0), 1.0f, 0);
        bgfx::setViewRect(0, 0, 0, bgfx.resolution.width, bgfx.resolution.height);
    }

    void Application::run() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            bgfx::touch(0);
            bgfx::frame();
        }
    }

    void Application::cleanup() {
        bgfx::shutdown();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
}