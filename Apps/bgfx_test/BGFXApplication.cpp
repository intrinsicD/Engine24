//
// Created by alex on 24.07.24.
//

#include "Application.h"


#if defined(_WIN32) || defined(_WIN64)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#endif

#include "bgfx/bgfx.h"
#include "Logger.h"
#include "Color.h"
#include "GLFW/glfw3.h"
#include "GLFW/glfw3native.h"
#include "imgui/imgui.h"

#undef Success

#include "Input.h"

namespace Bcg {
    static GLFWwindow *window;

    static void glfw_error_callback(int error, const char *description) {
        std::string message = "GLFW Error " + std::to_string(error) + ", " + description + "\n";
        Log::Error(message.c_str());
    }

    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
        Input::set_keyboard(window, key, scancode, action, mode);
    }

    static void mouse_cursor_callback(GLFWwindow *window, double xpos, double ypos) {
        Input::set_mouse_cursor_position(window, xpos, ypos);
    }

    static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
        Input::set_mouse_button(window, button, action, mods);
    }

    static void mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset) {
        Input::set_mouse_scrolling(window, xoffset, yoffset);
    }

    static void resize_callback(GLFWwindow *window, int width, int height) {
        bgfx::reset(width, height, BGFX_RESET_VSYNC);
    }

    static void close_callback(GLFWwindow *window) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    static void drop_callback(GLFWwindow *window, int count, const char **paths) {

    }

    Application::Application() {
        Engine::Context().emplace<bgfx::Init>();
    }

    void Application::init(int width, int height, const char *title) {
        auto &bgfx =  Engine::Context().get<bgfx::Init>();
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
        glfwSetKeyCallback(window, key_callback);
        glfwSetCursorPosCallback(window, mouse_cursor_callback);
        glfwSetMouseButtonCallback(window, mouse_button_callback);
        glfwSetScrollCallback(window, mouse_scrolling);
        glfwSetWindowCloseCallback(window, close_callback);
        glfwSetWindowSizeCallback(window, resize_callback);
        glfwSetDropCallback(window, drop_callback);

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
        imguiCreate();

        bgfx::setViewClear(0, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, floatColorToUint32(0.2, 0.3, 0.3, 1.0), 1.0f, 0);
        bgfx::setViewRect(0, 0, 0, bgfx.resolution.width, bgfx.resolution.height);

        float xscale, yscale;
        glfwGetWindowContentScale(window, &xscale, &yscale);
        float dpi_scale = (xscale + yscale) / 2.0f;
        ImGuiIO &io = ImGui::GetIO();
        io.FontGlobalScale = dpi_scale;
    }

    void Application::run() {
        auto &bgfx =  Engine::Context().get<bgfx::Init>();
        auto &mouse = Engine::Context().emplace<Mouse>();
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            imguiBeginFrame(mouse.cursor.current.ssp.x(), mouse.cursor.current.ssp.y(),
                            (mouse.left() ? IMGUI_MBUT_LEFT : 0)
                            | (mouse.right() ? IMGUI_MBUT_RIGHT : 0)
                            | (mouse.middle() ? IMGUI_MBUT_MIDDLE : 0), mouse.scroll_offset.y(), bgfx.resolution.width,
                            bgfx.resolution.height
            );

            ImGui::BeginMainMenuBar();
            if (ImGui::BeginMenu("Test")) {
                ImGui::EndMenu();
            }

            ImGui::EndMainMenuBar();

            imguiEndFrame();

            bgfx::touch(0);
            bgfx::frame();
        }
    }

    void Application::cleanup() {
        imguiDestroy();

        bgfx::shutdown();

        glfwDestroyWindow(window);
        glfwTerminate();
    }
}