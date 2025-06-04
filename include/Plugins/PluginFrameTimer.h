//
// Created by alex on 10.07.24.
//

#ifndef ENGINE24_PLUGINFRAMETIMER_H
#define ENGINE24_PLUGINFRAMETIMER_H

#include "Plugin.h"
#include "Timer.h"


namespace Bcg {
    class PluginFrameTimer : public Plugin {
    public:
        PluginFrameTimer();

        ~PluginFrameTimer() override = default;

        static float delta();

        void activate() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;
    };
}

#endif //ENGINE24_PLUGINFRAMETIMER_H
