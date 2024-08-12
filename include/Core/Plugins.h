//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_PLUGINS_H
#define ENGINE24_PLUGINS_H

#include <memory>
#include <string>
#include "Plugin.h"

namespace Bcg {
    struct Plugins {
        static void init();

        static void init_user_plugin(const std::string &name);

        static void add_plugin(std::unique_ptr<Plugin> uptr);

        static void activate_all();

        static void begin_frame_all();

        static void update_all();

        static void render_all();

        static void render_menu();

        static void render_gui();

        static void end_frame();

        static void deactivate_all();
    };
}

#endif //ENGINE24_PLUGINS_H
