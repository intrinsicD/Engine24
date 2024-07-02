//
// Created by alex on 19.06.24.
//

#include "Mouse.h"
#include "GLFW/glfw3.h"

namespace Bcg {
    bool Mouse::left() const {
        return pressed[GLFW_MOUSE_BUTTON_1];
    }

    bool Mouse::middle() const {
        return pressed[GLFW_MOUSE_BUTTON_2];
    }

    bool Mouse::right() const {
        return pressed[GLFW_MOUSE_BUTTON_3];
    }

    bool Mouse::any() const { return left() || middle() || right(); }
}