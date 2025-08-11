//
// Created by alex on 10.07.24.
//

#include "PluginFrameTimer.h"
#include "FrameTimerGui.h"
#include "../../Graphics/ModuleGraphics.h"
#include "Logger.h"
#include "Engine.h"
#include "imgui.h"

namespace Bcg {
    PluginFrameTimer::PluginFrameTimer() : Plugin("FrameTimer") {
    }

    float PluginFrameTimer::delta() {
        return Engine::Context().get<FrameTimer>().timer.delta;
    }

    void PluginFrameTimer::activate() {
        if (base_activate()) {
            Engine::Context().emplace<Timer>();
            Engine::Context().emplace<FrameTimer>();
        }
    }

    void PluginFrameTimer::end_frame() {
        auto &frame = Engine::Context().get<FrameTimer>();
        frame.timer.update();
        frame.avg_frame_time = frame.avg_frame_time * frame.frame_counter + frame.timer.delta;
        frame.avg_frame_time /= ++frame.frame_counter;
        frame.fps = 1. / frame.avg_frame_time;
    }

    void PluginFrameTimer::deactivate() {
        if (base_deactivate()) {
            auto &timer = Engine::Context().get<Timer>();
            auto &frame = Engine::Context().get<FrameTimer>();

            Log::Info("Average Frame Time: " + std::to_string(frame.avg_frame_time) + " s");
            Log::Info("Average Frames Per Seconds: " + std::to_string(frame.fps));
            Log::Info("Number of Total Frames: " + std::to_string(frame.frame_counter));
            Log::Info("Total Runtime:  " + std::to_string(timer.delta) + " s");
        }
    }

    static bool show_gui = false;

    void PluginFrameTimer::render_menu() {
        if (ImGui::BeginMenu("Rendering")) {
            ImGui::MenuItem(name.c_str(), nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginFrameTimer::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name.c_str(), &show_gui, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar)) {
                auto &timer = Engine::Context().get<Timer>();
                auto &frame = Engine::Context().get<FrameTimer>();
                Gui::Show(frame);
                ImGui::Text("Total Runtime:  %f s", timer.stop().delta);
                ImGui::End();
            }
        }
    }
}
