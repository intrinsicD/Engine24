//
// Created by alex on 30.07.24.
//

#ifndef ENGINE24_PLUGINCUDA_H
#define ENGINE24_PLUGINCUDA_H

#include "Plugin.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class PluginCuda : public Plugin {
    public:
        PluginCuda() : Plugin("Cuda") {}

        ~PluginCuda() override = default;

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

#endif //ENGINE24_PLUGINCUDA_H
