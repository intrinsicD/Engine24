//
// Created by alex on 19.07.24.
//

#include "CoordinateSystemsGui.h"
#include "imgui.h"

namespace Bcg::Gui{
    void Show(const Points &points ){
        ImGui::Text("ScreenSpacePos: %lf, %lf", points.ssp.x(), points.ssp.y());
        ImGui::Text("ScreenSpacePosDpiAdjusted: %lf, %lf", points.sspda.x(), points.sspda.y());
        ImGui::Text("NdcSpacePos: %lf, %lf, %lf", points.ndc.x(), points.ndc.y(), points.ndc.z());
        ImGui::Text("ViewSpacePos: %lf, %lf, %lf", points.vsp.x(), points.vsp.y(), points.vsp.z());
        ImGui::Text("WorldSpacePos: %lf, %lf, %lf", points.wsp.x(), points.wsp.y(), points.wsp.z());
    }
}