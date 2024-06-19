//
// Created by alex on 19.06.24.
//

#include "Keybaord.h"
#include "GLFW/glfw3.h"

namespace Bcg{
    bool Keyboard::shift() const{
        return pressed[GLFW_KEY_LEFT_SHIFT] || pressed[GLFW_KEY_RIGHT_SHIFT];
    }

    bool Keyboard::strg() const {
        return pressed[GLFW_KEY_LEFT_CONTROL] || pressed[GLFW_KEY_RIGHT_CONTROL];
    }

    bool Keyboard::alt() const {
        return pressed[GLFW_KEY_LEFT_ALT] || pressed[GLFW_KEY_RIGHT_ALT];
    }

    bool Keyboard::esc() const{
        return pressed[GLFW_KEY_ESCAPE];
    }
}