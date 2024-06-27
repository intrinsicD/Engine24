//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_PLUGINOPENGLRENDERER_H
#define ENGINE24_PLUGINOPENGLRENDERER_H


#include "Plugin.h"
#include <vector>
#include <string>
#include "MatVec.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class PluginOpenGLRenderer : public Plugin {
    public:
        PluginOpenGLRenderer();

        ~PluginOpenGLRenderer() override = default;

        static void unhide_entity(entt::entity entity_id);

        static void hide_entity(entt::entity entity_id);

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

#endif //ENGINE24_PLUGINOPENGLRENDERER_H
