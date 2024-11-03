//
// Created by alex on 11/3/24.
//

#ifndef PLUGINPOINTCLOUDSAMPLING_H
#define PLUGINPOINTCLOUDSAMPLING_H

#include "Plugin.h"

namespace Bcg {
    class PluginPointCloudSampling : public Plugin {
    public:
        PluginPointCloudSampling();

        ~PluginPointCloudSampling() override = default;

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

#endif //PLUGINPOINTCLOUDSAMPLING_H
