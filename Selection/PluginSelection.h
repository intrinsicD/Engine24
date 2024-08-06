//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_PLUGINSELECTION_H
#define ENGINE24_PLUGINSELECTION_H

#include "Plugin.h"

namespace Bcg {
    class PluginSelection : public Plugin {
    public:
        PluginSelection();

        ~PluginSelection() = default;

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

#endif //ENGINE24_PLUGINSELECTION_H
