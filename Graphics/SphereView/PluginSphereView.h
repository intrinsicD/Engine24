//
// Created by alex on 02.08.24.
//

#ifndef ENGINE24_PLUGINSPHEREVIEW_H
#define ENGINE24_PLUGINSPHEREVIEW_H

#include "Plugin.h"

namespace Bcg {
    class PluginSphereView : public Plugin {
    public:
        PluginSphereView() : Plugin("SphereView") {}

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

#endif //ENGINE24_PLUGINSPHEREVIEW_H
