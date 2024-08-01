//
// Created by alex on 01.08.24.
//

#ifndef ENGINE24_PLUGINKDTREE_H
#define ENGINE24_PLUGINKDTREE_H

#include "Plugin.h"

namespace Bcg {
    class PluginKDTree : public Plugin {
    public:
        PluginKDTree() : Plugin("KDTree") {

        }

        ~PluginKDTree() override = default;

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

#endif //ENGINE24_PLUGINKDTREE_H
