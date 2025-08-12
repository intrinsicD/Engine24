//
// Created by alex on 30.07.24.
//

#ifndef ENGINE24_PLUGINPOINTCLOUD_H
#define ENGINE24_PLUGINPOINTCLOUD_H


#include "Plugin.h"
#include "PointCloud.h"
#include "Command.h"


namespace Bcg {
    class PluginPointCloud : public Plugin {
    public:
        PluginPointCloud();

        ~PluginPointCloud() override = default;

        static PointCloud load(const std::string &path);

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };

    namespace Commands {

    }
}
#endif //ENGINE24_PLUGINPOINTCLOUD_H
