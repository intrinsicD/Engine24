//
// Created by alex on 19.06.24.
//

#ifndef ENGINE24_INPUT_H
#define ENGINE24_INPUT_H

#include "Plugin.h"
#include "Keybaord.h"
#include "Mouse.h"

namespace Bcg {
    class Input : public Plugin {
    public:
        Input();

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        static void render_gui(const Keyboard &keyboard);

        static void render_gui(const Mouse::Cursor &cursor);

        static void render_gui(const Mouse &mouse);

        void render() override;
    };
}

#endif //ENGINE24_INPUT_H
