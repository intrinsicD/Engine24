//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_INPUT_H
#define ENGINE24_INPUT_H

#include "Plugin.h"
#include "Keyboard.h"
#include "Mouse.h"

struct GLFWwindow;

namespace Bcg {
    class Input : public Plugin {
    public:
        Input();

        static Keyboard &set_keyboard(GLFWwindow *window, int key, int scancode, int action, int mode);

        static Mouse &set_mouse_cursor_position(GLFWwindow *window, double xpos, double ypos);

        static Mouse &set_mouse_button(GLFWwindow *window, int button, int action, int mods);

        static Mouse &set_mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset);

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };
}

#endif //ENGINE24_INPUT_H
