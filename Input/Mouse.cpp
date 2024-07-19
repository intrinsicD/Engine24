//
// Created by alex on 19.06.24.
//

#include "Mouse.h"
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "Graphics.h"
#include "Engine.h"
#include "Camera.h"

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
            auto &camera = Engine::Context().get<Camera>();
            Points points = PointTransformer(Graphics::dpi_scaling(), Graphics::get_viewport_dpi_adjusted(), camera.proj,
                                             camera.view).apply(cursor.raw.pos);
            ImGui::Text("ScreenSpacePos: %lf, %lf", points.ssp.x(), points.ssp.y());
            ImGui::Text("ScreenSpacePosDpiAdjusted: %lf, %lf", points.sspda.x(), points.sspda.y());
            ImGui::Text("NdcSpacePos: %lf, %lf, %lf", points.ndc.x(), points.ndc.y(), points.ndc.z());
            ImGui::Text("ViewSpacePos: %lf, %lf, %lf", points.vsp.x(), points.vsp.y(), points.vsp.z());
            ImGui::Text("WorldSpacePos: %lf, %lf, %lf", points.wsp.x(), points.wsp.y(), points.wsp.z());
        }
    }
}