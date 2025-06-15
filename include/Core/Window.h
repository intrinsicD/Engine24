//
// Created by alex on 6/15/25.
//

#ifndef WINDOW_H
#define WINDOW_H

#include "InputManager.h"
#include "MatVec.h"
#include <string>

struct GLFWwindow;

namespace Bcg {
    class Window {
    public:
        Window(int width, int height, const char *title, InputManager &input_manager);

        ~Window();

        bool exists() const {
            return m_window != nullptr;
        }

        bool should_close() const;

        void poll_events();

        void swap_buffers();

        GLFWwindow *get_native_window() const {
            return m_window;
        }

        void set_title(const std::string &title);

        float get_aspect_ratio() const;

        Vector<float, 2> get_xy_dpi_scaling() const;

        Vector<int, 2> get_window_size() const;

        Vector<int, 2> get_window_pos() const;

        Vector<int, 2> get_framebuffer_size() const;

    private:
        void setup_callbacks();

        static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);

        static void mouse_cursor_callback(GLFWwindow *window, double xpos, double ypos);

        static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods);

        static void mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset);

        static void window_resize_callback(GLFWwindow *window, int width, int height);

        static void framebuffer_resize_callback(GLFWwindow *window, int width, int height);

        static void close_callback(GLFWwindow *window);

        static void drop_callback(GLFWwindow *window, int count, const char **paths);

        GLFWwindow *m_window = nullptr;
        InputManager &m_input_manager;
    };
}

#endif //WINDOW_H
