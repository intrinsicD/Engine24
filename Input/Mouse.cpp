//
// Created by alex on 19.06.24.
//

#include "Mouse.h"
#include "imgui.h"
#include "CoordinateSystemsGui.h"

namespace Bcg {
    bool Mouse::left() const {
        return pressed[Mouse::ButtonType::Left];
    }

    bool Mouse::middle() const {
        return pressed[Mouse::ButtonType::Middle];
    }

    bool Mouse::right() const {
        return pressed[Mouse::ButtonType::Right];
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
            for (const auto button: mouse.current_buttons) {
                ImGui::Text("%d", button);
                ImGui::SameLine();
            }
            ImGui::Text("}");
        }

        void Show(const Mouse::Cursor &cursor) {
            if(ImGui::CollapsingHeader("Current")){
                Show(cursor.current);
            }
            if(ImGui::CollapsingHeader("Last Left")){
                ImGui::Text("Press");
                ImGui::Separator();
                Show(cursor.last_left.press);
                ImGui::Separator();
                ImGui::Text("Release");
                ImGui::Separator();
                Show(cursor.last_left.release);
                ImGui::Separator();
            }
            if(ImGui::CollapsingHeader("Last Middle")){
                ImGui::Text("Press");
                ImGui::Separator();
                Show(cursor.last_middle.press);
                ImGui::Separator();
                ImGui::Text("Release");
                ImGui::Separator();
                Show(cursor.last_middle.release);
                ImGui::Separator();
            }
            if(ImGui::CollapsingHeader("Last Right")){
                ImGui::Text("Press");
                ImGui::Separator();
                Show(cursor.last_right.press);
                ImGui::Separator();
                ImGui::Text("Release");
                ImGui::Separator();
                Show(cursor.last_right.release);
                ImGui::Separator();
            }
        }
    }
}