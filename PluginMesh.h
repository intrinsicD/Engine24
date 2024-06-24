//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGINMESH_H
#define ENGINE24_PLUGINMESH_H

#include "Plugin.h"
#include <vector>
#include <string>
#include <unordered_map>
#include "MeshComponent.h"

namespace Bcg {
    class PluginMesh : public Plugin {
    public:
        PluginMesh();

        ~PluginMesh() override = default;

        MeshComponent load(const char *path);

        MeshComponent load_obj(const char *path);

        MeshComponent load_off(const char *path);

        MeshComponent load_stl(const char *path);

        MeshComponent load_ply(const char *path);

        void merge_vertices(MeshComponent &mesh, float tol);

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

#endif //ENGINE24_PLUGINMESH_H
