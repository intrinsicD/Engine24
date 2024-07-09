//
// Created by alex on 05.07.24.
//

#include "MeshViewer.h"
#include "Engine.h"
#include "Graphics.h"
#include "Plugins.h"
#include "Timer.h"
#include "Logger.h"

namespace Bcg {
    MeshViewer::MeshViewer() {

    }

    void MeshViewer::run() {
        Bcg::Engine engine;
        if (!Bcg::Graphics::init()) {
            return;
        }
        Bcg::Graphics::set_window_title("MeshViewer");

        Bcg::Plugins::init();
        Bcg::Plugins::activate_all();
        Bcg::Engine::ExecuteCmdBuffer();

        // Game loop
        auto timer = Timer();
        float avg_frame_time = 0;
        size_t frame_counter = 0;
        auto &frame_timer = Engine::Context().emplace<FrameTimer>();
        while (!Bcg::Graphics::should_close()) {
            {
                Bcg::Graphics::poll_events();
                Bcg::Plugins::begin_frame_all();
                Bcg::Plugins::update_all();
                Bcg::Engine::ExecuteCmdBuffer();
            }
            {
                Bcg::Graphics::clear_framebuffer();
                Bcg::Plugins::render_all();
                Bcg::Engine::ExecuteCmdBuffer();
                Bcg::Graphics::start_gui();
                Bcg::Plugins::render_menu();
                Bcg::Plugins::render_gui();
                Bcg::Graphics::render_menu();
                Bcg::Graphics::render_gui();
                Bcg::Graphics::end_gui();
                Bcg::Engine::ExecuteCmdBuffer();
                Bcg::Plugins::end_frame();
                Bcg::Graphics::swap_buffers();
            }
            frame_timer.update();
            avg_frame_time = avg_frame_time * frame_counter + frame_timer.delta;
            avg_frame_time /= ++frame_counter;
        }
        Engine::Context().erase<FrameTimer>();
        Bcg::Plugins::deactivate_all();
        Log::Info("Average Frame Time: " + std::to_string(avg_frame_time));
        Log::Info("Average Frames Per Seconds: " + std::to_string(1. / avg_frame_time));
        Log::Info("Number of Total Frames:  " + std::to_string(frame_counter));
    }
}