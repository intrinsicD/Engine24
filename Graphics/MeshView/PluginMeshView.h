//
// Created by alex on 04.08.24.
//

#ifndef ENGINE24_PLUGINMESHVIEW_H
#define ENGINE24_PLUGINMESHVIEW_H

#include "Plugin.h"

namespace Bcg {
    class PluginMeshView : public Plugin {
    public:
        PluginMeshView() : Plugin("MeshView") {}

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
#endif //ENGINE24_PLUGINMESHVIEW_H
