//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_PLUGINAABB_H
#define ENGINE24_PLUGINAABB_H

#include "Plugin.h"
#include "AABB.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class PluginAABB : public Plugin {
    public:
        explicit PluginAABB();

        ~PluginAABB() override = default;

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

#endif //ENGINE24_PLUGINAABB_H
