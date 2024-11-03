//
// Created by alex on 11/3/24.
//

#include "MouseGui.h"
#include "imgui.h"
#include "CoordinateSystemsGui.h"

namespace Bcg::Gui {
    void Show(const Mouse &mouse) {
        ImGui::Text("Left: %d", mouse.left());
        ImGui::Text("Middle: %d", mouse.middle());
        ImGui::Text("Right: %d", mouse.right());
        ImGui::Text("Scrolling: %d", mouse.scrolling);
        ImGui::Text("Scroll Offset: %lf, %lf", mouse.scroll_offset.x, mouse.scroll_offset.y);
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
        if (ImGui::CollapsingHeader("Current")) {
            Show(cursor.current);
        }
        if (ImGui::CollapsingHeader("Last Left")) {
            ImGui::Text("Press");
            ImGui::Separator();
            Show(cursor.last_left.press);
            ImGui::Separator();
            ImGui::Text("Release");
            ImGui::Separator();
            Show(cursor.last_left.release);
            ImGui::Separator();
        }
        if (ImGui::CollapsingHeader("Last Middle")) {
            ImGui::Text("Press");
            ImGui::Separator();
            Show(cursor.last_middle.press);
            ImGui::Separator();
            ImGui::Text("Release");
            ImGui::Separator();
            Show(cursor.last_middle.release);
            ImGui::Separator();
        }
        if (ImGui::CollapsingHeader("Last Right")) {
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
