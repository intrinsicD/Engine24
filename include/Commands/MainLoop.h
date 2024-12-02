//
// Created by alex on 29.11.24.
//

#ifndef ENGINE24_MAINLOOP_H
#define ENGINE24_MAINLOOP_H

#include "CommandDoubleBuffer.h"
#include "EventsMain.h"
#include "entt/signal/dispatcher.hpp"

namespace Bcg::Commands {
    using InitializationCommands = DoubleCommandBuffer;
    using StartupCommands = DoubleCommandBuffer;
    using ShutdownCommands = DoubleCommandBuffer;

    struct MainLoop {
        DoubleCommandBuffer begin_loop;     // Engine initialization
        DoubleCommandBuffer prepare_scene;  // Combines begin_scene + update_scene
        DoubleCommandBuffer end_scene;      // Scene cleanup
        DoubleCommandBuffer prepare_render; // Combines begin_render + render_scene
        DoubleCommandBuffer end_render;     // Render cleanup
        DoubleCommandBuffer render_gui;     // Combines begin_gui, render_gui, end_gui
        DoubleCommandBuffer end_loop;       // Final loop tasks

        void handle(entt::dispatcher &dispatcher) {
            dispatcher.trigger<Events::Synchronize>();
            begin_loop.handle();
            prepare_scene.handle();
            end_scene.handle();
            prepare_render.handle();
            end_render.handle();
            render_gui.handle();
            end_loop.handle();
        }
    };
}

#endif //ENGINE24_MAINLOOP_H
