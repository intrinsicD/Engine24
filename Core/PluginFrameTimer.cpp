//
// Created by alex on 10.07.24.
//

#include "PluginFrameTimer.h"
#include "Timer.h"
#include "Logger.h"
#include "Engine.h"
#include "imgui.h"

namespace Bcg {
    struct FrameTimer {
        Timer timer;
        float avg_frame_time = 0;
        float fps = 0;
        size_t frame_counter = 0;
    };

    PluginFrameTimer::PluginFrameTimer() : Plugin("FrameTimer") {}

    float PluginFrameTimer::delta() {
        return Engine::Context().get<FrameTimer>().timer.delta;
    }

    void PluginFrameTimer::activate() {
        Engine::Context().emplace<Timer>();
        Engine::Context().emplace<FrameTimer>();
        Plugin::activate();
    }

    void PluginFrameTimer::begin_frame() {}

    void PluginFrameTimer::update() {}

    void PluginFrameTimer::end_frame() {
        auto &frame = Engine::Context().get<FrameTimer>();
        frame.timer.update();
        frame.avg_frame_time = frame.avg_frame_time * frame.frame_counter + frame.timer.delta;
        frame.avg_frame_time /= ++frame.frame_counter;
        frame.fps = 1. / frame.avg_frame_time;
    }

    void PluginFrameTimer::deactivate() {
        auto &timer = Engine::Context().get<Timer>();
        auto &frame = Engine::Context().get<FrameTimer>();
        Log::Info("Average Frame Time: " + std::to_string(frame.avg_frame_time) + " s");
        Log::Info("Average Frames Per Seconds: " + std::to_string(frame.fps));
        Log::Info("Number of Total Frames: " + std::to_string(frame.frame_counter));
        Log::Info("Total Runtime:  " + std::to_string(timer.delta) + " s");
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginFrameTimer::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            ImGui::MenuItem(name, nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginFrameTimer::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name, &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &timer = Engine::Context().get<Timer>();
                auto &frame = Engine::Context().get<FrameTimer>();
                ImGui::Text("Average Frame Time: %f s", frame.avg_frame_time);
                ImGui::Text("Average Frames Per Seconds: %f", frame.fps);
                ImGui::Text("Number of Total Frames:  %zu", frame.frame_counter);
                ImGui::Text("Total Runtime:  %f s", timer.stop().delta);
                ImGui::End();
            }
        }
    }

    void PluginFrameTimer::render() {}
}