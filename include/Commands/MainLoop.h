//
// Created by alex on 29.11.24.
//

#ifndef ENGINE24_MAINLOOP_H
#define ENGINE24_MAINLOOP_H

#include "Command.h"

namespace Bcg::Commands {
    using InitializationCommands = CompositeCommand;
    using StartupCommands = CompositeCommand;
    using ShutdownCommands = CompositeCommand;

    struct MainLoop {
        CompositeCommand begin_loop{"begin_loop"};     // Engine initialization
        CompositeCommand prepare_scene{"prepare_scene"};  // Combines begin_scene + update_scene
        CompositeCommand end_scene{"end_scene"};      // Scene cleanup
        CompositeCommand prepare_render{"prepare_render"}; // Combines begin_render + render_scene
        CompositeCommand end_render{"end_render"};     // Render cleanup
        CompositeCommand render_gui{"render_gui"};     // Combines begin_gui, render_gui, end_gui
        CompositeCommand end_loop{"end_loop"};       // Final loop tasks

        //TODO ensure thread safety and introduce double buffering (vector swapping)
        void execute() const {
            begin_loop.execute();
            prepare_scene.execute();
            end_scene.execute();
            prepare_render.execute();
            end_render.execute();
            render_gui.execute();
            end_loop.execute();
        }
    };
}

#endif //ENGINE24_MAINLOOP_H
