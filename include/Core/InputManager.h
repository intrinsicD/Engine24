//
// Created by alex on 6/15/25.
//

#ifndef INPUTMANAGER_H
#define INPUTMANAGER_H

struct GLFWwindow;

namespace Bcg {
    class InputManager {
    public:
        InputManager();

        ~InputManager();

        void handle_key_callback(GLFWwindow *window, int key, int scancode, int action, int mods);

        void handle_mouse_button_callback(GLFWwindow *window, int button, int action, int mods);

        void handle_mouse_cursor_callback(GLFWwindow *window, double xpos, double ypos);

        void handle_mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset);

        void handle_window_resize_callback(GLFWwindow *window, int width, int height);

        void handle_framebuffer_resize_callback(GLFWwindow *window, int width, int height);

        void handle_close_callback(GLFWwindow *window);

        void handle_drop_callback(GLFWwindow *window, int count, const char **paths);

    };
}

#endif //INPUTMANAGER_H
