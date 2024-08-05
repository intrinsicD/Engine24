//
// Created by alex on 05.08.24.
//

#ifndef ENGINE24_PLUGINVECTORFIELDVIEWS_H
#define ENGINE24_PLUGINVECTORFIELDVIEWS_H

#include "Plugin.h"

namespace Bcg {
    class PluginVectorfieldViews : public Plugin {
    public:
        PluginVectorfieldViews() : Plugin("PluginVectorfieldViews") {}

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
#endif //ENGINE24_PLUGINVECTORFIELDVIEWS_H
