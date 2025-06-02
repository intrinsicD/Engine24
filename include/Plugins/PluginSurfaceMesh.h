//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGINSURFACEMESH_H
#define ENGINE24_PLUGINSURFACEMESH_H

#include "Plugin.h"
#include "SurfaceMesh.h"
#include "Command.h"

namespace Bcg {
    class PluginSurfaceMesh : public Plugin {
    public:
        PluginSurfaceMesh();

        ~PluginSurfaceMesh() override = default;

        static SurfaceMesh read(const std::string &filepath);

        static bool write(const std::string &filepath, const SurfaceMesh &mesh);

        static void merge_vertices(SurfaceMesh &mesh, float tol);

        void activate() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

    };
}

#endif //ENGINE24_PLUGINSURFACEMESH_H
