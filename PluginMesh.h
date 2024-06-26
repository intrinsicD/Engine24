//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGINMESH_H
#define ENGINE24_PLUGINMESH_H

#include "Plugin.h"
#include <vector>
#include <string>
#include <unordered_map>
#include "pmp/surface_mesh.h"
#include "pmp/algorithms/curvature.h"

namespace Bcg {
    class PluginMesh : public Plugin {
    public:
        PluginMesh();

        ~PluginMesh() override = default;

        static pmp::SurfaceMesh load(const char *path);

        static pmp::SurfaceMesh load_obj(const char *path);

        static pmp::SurfaceMesh load_off(const char *path);

        static pmp::SurfaceMesh load_stl(const char *path);

        static pmp::SurfaceMesh load_ply(const char *path);

        static pmp::SurfaceMesh load_pmp(const char *path);

        static void merge_vertices(pmp::SurfaceMesh &mesh, float tol);

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
