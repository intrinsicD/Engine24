//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_EVENTSCALLBACKS_H
#define ENGINE24_EVENTSCALLBACKS_H

struct GLFWwindow;

namespace Bcg::Events::Callback {
    struct Key {
        GLFWwindow *window;
        int key, scancode, action, mode;
    };

    struct MouseCursor {
        GLFWwindow *window;
        double xpos, ypos;
    };

    struct MouseButton {
        GLFWwindow *window;
        int button, action, mods;
    };

    struct MouseScroll {
        GLFWwindow *window;
        double xoffset, yoffset;
    };

    struct WindowResize {
        GLFWwindow *window;
        int width, height;
    };

    struct WindowClose {
        GLFWwindow *window;
    };

    struct Drop {
        GLFWwindow *window;
        int count;
        const char **paths;
    };
}

#endif //ENGINE24_EVENTSCALLBACKS_H
