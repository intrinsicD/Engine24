//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_PLUGINPICKER_H
#define ENGINE24_PLUGINPICKER_H

#include "Picker.h"
#include "Plugin.h"
#include "CoordinateSystems.h"

namespace Bcg{
    class PluginPicker : public Plugin {
    public:
        PluginPicker();

        ~PluginPicker() override = default;

        static Picked &pick(const ScreenSpacePos &pos);

        static Picked &last_picked();

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

#endif //ENGINE24_PLUGINPICKER_H
