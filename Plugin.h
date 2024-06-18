//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGIN_H
#define ENGINE24_PLUGIN_H

namespace Bcg {
    class Plugin {
    public:
        virtual ~Plugin() = default;

        virtual void activate() = 0;

        virtual void update() = 0;

        virtual void deactivate() = 0;

        virtual void render_menu() = 0;

        virtual void render_gui() = 0;

        virtual void render() = 0;
    };
}

#endif //ENGINE24_PLUGIN_H
