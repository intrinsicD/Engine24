//
// Created by alex on 23.07.24.
//

#include "PluginWindow.h"
#include "Engine.h"
#include "Logger.h"
#include "EventsCallbacks.h"
#include "Mouse.h"
#include "Keyboard.h"
#include "GLFW/glfw3.h"

namespace Bcg {
    static void glfw_error_callback(int error, const char *description) {
        Log::Error("GLFW Error " + std::to_string(error) + ", " + description + "\n");
    }

    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
        auto &keyboard = Input::set_keyboard(window, key, scancode, action, mode);

        if (!keyboard.gui_captured) {
            auto &dispatcher = Engine::Dispatcher();

            dispatcher.trigger<Events::Callback::Key>({window, key, scancode, action, mode});
            if (keyboard.esc()) {
                glfwSetWindowShouldClose(window, true);
            }
            handle(key, action, dispatcher);
        }
    }

    static void mouse_cursor_callback(GLFWwindow *window, double xpos, double ypos) {
        Input::set_mouse_cursor_position(window, xpos, ypos);

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::MouseCursor>({window, xpos, ypos});
        }
    }

    static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
        Input::set_mouse_button(window, button, action, mods);

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::MouseButton>({window, button, action, mods});
        }
    }

    static void mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset) {
        Input::set_mouse_scrolling(window, xoffset, yoffset);

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::MouseScroll>({window, xoffset, yoffset});
        }
    }

    static void resize_callback(GLFWwindow *window, int width, int height) {
        glViewport(0, 0, width, height);

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::WindowResize>({window, width, height});
        }
    }

    static void close_callback(GLFWwindow *window) {
        glfwSetWindowShouldClose(window, true);

        Engine::Dispatcher().trigger<Events::Callback::WindowClose>({window});
    }

    static void drop_callback(GLFWwindow *window, int count, const char **paths) {
        for (int i = 0; i < count; ++i) {
            Log::Info("Dropped: " + std::string(paths[i]));
        }

        Engine::Dispatcher().trigger<Events::Callback::Drop>({window, count, paths});
    }

    inline GLFWwindow *GLFW(void *handle) {
        return static_cast<GLFWwindow *>(handle);
    }

    bool PluginWindow::init(WindowInit &user_init) {
        glfwSetErrorCallback(glfw_error_callback);

        if (!glfwInit()) {
            return false;
        }

        return true;
    }

    void *PluginWindow::create(int width, int height, const char *title) {}

    void PluginWindow::make_current(void *handle) {
        glfwMakeContextCurrent(GLFW(handle));
    }

    void *PluginWindow::get_current() {
        return glfwGetCurrentContext();
    }

    bool PluginWindow::destroy(void *handle) {}

    const char *PluginWindow::get_title(void *handle) {}

    void PluginWindow::set_title(void *handle, const char *title) {
        glfwSetWindowTitle(GLFW(handle), title);
    }

    void PluginWindow::poll_events() { glfwPollEvents(); }

    void PluginWindow::swap_buffers(void *handle) { glfwSwapBuffers(GLFW(handle)); }

    void PluginWindow::swap_buffers() { glfwSwapBuffers(glfwGetCurrentContext()); }

    bool PluginWindow::should_close(void *handle) { return glfwWindowShouldClose(GLFW(handle)); }

    void PluginWindow::shutdown() { glfwTerminate(); }

    float PluginWindow::get_dpi_scaling(void *handle) {
        float dpi_scaling_factor;
        glfwGetWindowContentScale(GLFW(handle), &dpi_scaling_factor, &dpi_scaling_factor);
        return dpi_scaling_factor;
    }

    void PluginWindow::get_width_and_height(void *handle, int &width, int &height) {
        glfwGetWindowSize(GLFW(handle), &width, &height);
    }
}