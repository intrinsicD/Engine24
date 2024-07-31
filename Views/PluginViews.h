//
// Created by alex on 31.07.24.
//

#ifndef ENGINE24_PLUGINVIEWS_H
#define ENGINE24_PLUGINVIEWS_H

#include "Plugin.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class PluginViews : public Plugin {
    public:
        PluginViews() : Plugin("Views") {}

        ~PluginViews() override = default;

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };
}
#endif //ENGINE24_PLUGINVIEWS_H
