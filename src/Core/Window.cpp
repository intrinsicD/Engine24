//
// Created by alex on 6/15/25.
//

#include "Window.h"
#include "Logger.h"
#include "Engine.h"
#include "PluginInput.h"
#include "HandleGlfwKeyEvents.h"
#include "EventsCallbacks.h"
#include <GLFW/glfw3.h>

namespace Bcg {
    Window::Window(int width, int height, const char *title, InputManager &input_manager) : m_input_manager(
        input_manager) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        m_window = glfwCreateWindow(width, height, title, NULL, NULL);

        if (!m_window) {
            Log::Error("Failed to create GLFW window");
            return;
        }

        glfwMakeContextCurrent(m_window);
        glfwSwapInterval(1);
        glfwSetWindowUserPointer(m_window, this);

        setup_callbacks();
    }

    Window::~Window() {
        if (m_window) {
            glfwDestroyWindow(m_window);
            m_window = nullptr;
        }
    }

    bool Window::should_close() const {
        return glfwWindowShouldClose(m_window);
    }

    void Window::poll_events() {
        glfwPollEvents();
    }

    void Window::swap_buffers() {
        glfwSwapBuffers(m_window);
    }

    void Window::set_title(const std::string &title) {
        glfwSetWindowTitle(m_window, title.c_str());
    }

    float Window::get_aspect_ratio() const {
        Vector<int, 2> size = get_window_size();
        return size[0] / static_cast<float>(size[1]);
    }

    Vector<float, 2> Window::get_xy_dpi_scaling() const {
        Vector<float, 2> xy_scale;
        glfwGetWindowContentScale(m_window, &xy_scale[0], &xy_scale[1]);
        if (xy_scale[0] == 0 || xy_scale[1] == 0) {
            Log::Error("Failed to get DPI scaling, returning 1.0");
            return {1.0f, 1.0f}; // Fallback to default scaling
        }
        if (xy_scale[0] != xy_scale[1]) {
            Log::Info("DPI scaling is not uniform: xscale = {}, yscale = {}", xy_scale[0], xy_scale[1]);
        }
        return xy_scale;
    }

    Vector<int, 2> Window::get_window_size() const {
        Vector<int, 2> window_size;
        glfwGetWindowSize(m_window, &window_size[0], &window_size[1]);
        return window_size;
    }

    Vector<int, 2> Window::get_window_pos() const {
        Vector<int, 2> window_pos;
        glfwGetWindowPos(m_window, &window_pos[0], &window_pos[1]);
        return window_pos;
    }

    Vector<int, 2> Window::get_framebuffer_size() const {
        Vector<int, 2> framebuffer_size;
        glfwGetFramebufferSize(m_window, &framebuffer_size[0], &framebuffer_size[1]);
        return framebuffer_size;
    }

    void Window::key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
        Window *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        self->m_input_manager.handle_key_callback(window, key, scancode, action, mode);
    }

    void Window::mouse_cursor_callback(GLFWwindow *window, double xpos, double ypos) {
        Window *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        self->m_input_manager.handle_mouse_cursor_callback(window, xpos, ypos);
    }

    void Window::mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
        Window *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        self->m_input_manager.handle_mouse_button_callback(window, button, action, mods);
    }

    void Window::mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset) {
        Window *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        self->m_input_manager.handle_mouse_scrolling(window, xoffset, yoffset);
    }

    void Window::window_resize_callback(GLFWwindow *window, int width, int height) {
        Window *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        self->m_input_manager.handle_window_resize_callback(window, width, height);
    }

    void Window::framebuffer_resize_callback(GLFWwindow *window, int width, int height) {
        Window *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        self->m_input_manager.handle_framebuffer_resize_callback(window, width, height);
    }

    void Window::close_callback(GLFWwindow *window) {
        Window *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        self->m_input_manager.handle_close_callback(window);
    }

    void Window::drop_callback(GLFWwindow *window, int count, const char **paths) {
        Window *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
        self->m_input_manager.handle_drop_callback(window, count, paths);
    }

    void Window::setup_callbacks() {
        // Set the required callback functions
        glfwSetKeyCallback(m_window, key_callback);
        glfwSetCursorPosCallback(m_window, mouse_cursor_callback);
        glfwSetMouseButtonCallback(m_window, mouse_button_callback);
        glfwSetScrollCallback(m_window, mouse_scrolling);
        glfwSetWindowCloseCallback(m_window, close_callback);
        glfwSetWindowSizeCallback(m_window, window_resize_callback);
        glfwSetFramebufferSizeCallback(m_window, framebuffer_resize_callback);
        glfwSetDropCallback(m_window, drop_callback);
    }
}
