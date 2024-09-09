//
// Created by alex on 13.08.24.
//

#ifndef ENGINE24_PLUGINICP_H
#define ENGINE24_PLUGINICP_H

#include "Plugin.h"
#include "Command.h"

namespace Bcg {


    class PluginIcp : public Plugin {
    public:
        PluginIcp();

        ~PluginIcp() override = default;

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

#endif //ENGINE24_PLUGINICP_H
