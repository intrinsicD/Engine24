//
// Created by alex on 16.07.24.
//

#include "FrameTimerGui.h"
#include "imgui.h"

namespace Bcg::Gui{
    void Show(FrameTimer &frameTimer) {
        ImGui::Text("Average Frame Time: %f s", frameTimer.avg_frame_time);
        ImGui::Text("Average Frames Per Seconds: %f", frameTimer.fps);
        ImGui::Text("Number of Total Frames:  %zu", frameTimer.frame_counter);
    }
}