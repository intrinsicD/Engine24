//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGINMESH_H
#define ENGINE24_PLUGINMESH_H

#include "Plugin.h"
#include <vector>

namespace Bcg {
    struct MeshComponent {
        unsigned int vao;
        unsigned int vbo;
        unsigned int ebo;

        struct Material {
            unsigned int diffuse_texid;
        };

        std::vector<float> vertices;
        std::vector<unsigned int> indices;
        std::vector<Material> materials;
    };

    class PluginMesh : public Plugin {
    public:
        PluginMesh() = default;

        ~PluginMesh() override = default;

        MeshComponent load(const char *path);

        MeshComponent load_obj(const char *path);

        MeshComponent load_off(const char *path);

        MeshComponent load_stl(const char *path);

        MeshComponent load_ply(const char *path);

        void merge_vertices(MeshComponent &mesh, float tol);

        void activate() override;

        void update() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };
}

#endif //ENGINE24_PLUGINMESH_H
