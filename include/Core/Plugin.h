//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGIN_H
#define ENGINE24_PLUGIN_H

#include "entt/fwd.hpp"

namespace Bcg {
    class Plugin {
    public:
        explicit Plugin(const char *name);

        virtual ~Plugin() = default;

        virtual void activate();

        virtual void begin_frame() = 0;

        virtual void update() = 0;

        virtual void end_frame() = 0;

        virtual void deactivate();

        virtual void render_menu() = 0;

        virtual void render_gui() = 0;

        virtual void render() = 0;

        const char *name;
    };
}

#endif //ENGINE24_PLUGIN_H
