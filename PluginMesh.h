//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGINMESH_H
#define ENGINE24_PLUGINMESH_H

#include "Plugin.h"

namespace Bcg {
    struct MeshComponent {
        unsigned int vao;
        unsigned int vbo;
        unsigned int ebo;
    };

    class PluginMesh : public Plugin {
    public:
        PluginMesh() = default;

        ~PluginMesh() override = default;

        MeshComponent load(const char *path);

        void activate() override;

        void update() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };
}

#endif //ENGINE24_PLUGINMESH_H
