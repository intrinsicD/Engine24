//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_PLUGINOPENGLRENDERER_H
#define ENGINE24_PLUGINOPENGLRENDERER_H


#include "Plugin.h"
#include <vector>
#include <string>
#include "MatVec.h"

namespace Bcg {
    class PluginOpenGLRenderer : public Plugin {
    public:
        PluginOpenGLRenderer();

        ~PluginOpenGLRenderer() override = default;

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
