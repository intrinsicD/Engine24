//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGINSURFACEMESH_H
#define ENGINE24_PLUGINSURFACEMESH_H

#include "Plugin.h"
#include "SurfaceMesh.h"

namespace Bcg {
    class PluginSurfaceMesh : public Plugin {
    public:
        PluginSurfaceMesh();

        ~PluginSurfaceMesh() override = default;

        static SurfaceMesh read(const std::string &filepath);

        static bool write(const std::string &filepath, const SurfaceMesh &mesh);

        static void merge_vertices(SurfaceMesh &mesh, float tol);

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

#endif //ENGINE24_PLUGINSURFACEMESH_H
