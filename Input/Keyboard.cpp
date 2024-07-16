//
// Created by alex on 19.06.24.
//

#include "Keyboard.h"
#include "GLFW/glfw3.h"
#include "GLFWUtils.h"
#include "imgui.h"

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

    bool Keyboard::w() const{
        return pressed[GLFW_KEY_W];
    }

    bool Keyboard::a() const{
        return pressed[GLFW_KEY_A];
    }

    bool Keyboard::s() const{
        return pressed[GLFW_KEY_S];
    }

    bool Keyboard::d() const{
        return pressed[GLFW_KEY_D];
    }


    void Gui::Show(const Keyboard &keyboard) {
        ImGui::Text("Shift: %d", keyboard.shift());
        ImGui::Text("Strg: %d", keyboard.strg());
        ImGui::Text("Alt: %d", keyboard.alt());
        ImGui::Text("Esc: %d", keyboard.esc());
        ImGui::Text("Current Keys: {");
        ImGui::SameLine();
        for (const auto key: keyboard.current) {
            ImGui::Text("%s", KeyName(key));
            ImGui::SameLine();
        }
        ImGui::Text("}");
    }
}