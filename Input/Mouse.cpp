//
// Created by alex on 19.06.24.
//

#include "Mouse.h"
#include "GLFW/glfw3.h"
#include "imgui.h"

namespace Bcg {
    bool Mouse::left() const {
        return pressed[GLFW_MOUSE_BUTTON_1];
    }

    bool Mouse::middle() const {
        return pressed[GLFW_MOUSE_BUTTON_3];
    }

    bool Mouse::right() const {
        return pressed[GLFW_MOUSE_BUTTON_2];
    }

    bool Mouse::any() const { return left() || middle() || right(); }

    namespace Gui {
        void Show(const Mouse &mouse) {
            ImGui::Text("Left: %d", mouse.left());
            ImGui::Text("Middle: %d", mouse.middle());
            ImGui::Text("Right: %d", mouse.right());
            ImGui::Text("Scrolling: %d", mouse.scrolling);
            Show(mouse.cursor);
            ImGui::Text("Current Buttons: {");
            ImGui::SameLine();
            for (const auto button: mouse.current) {
                ImGui::Text("%d", button);
                ImGui::SameLine();
            }
            ImGui::Text("}");
        }

        void Show(const Mouse::Cursor &cursor) {
            ImGui::Text("Position: %lf, %lf", cursor.xpos, cursor.ypos);
        }
    }
}