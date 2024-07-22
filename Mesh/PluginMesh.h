//
// Created by alex on 18.06.24.
//

#ifndef ENGINE24_PLUGINMESH_H
#define ENGINE24_PLUGINMESH_H

#include "Plugin.h"
#include <vector>
#include <string>
#include <unordered_map>
#include "SurfaceMesh.h"
#include "Graphics.h"
#include "entt/entity/entity.hpp"

namespace Bcg {
    class PluginMesh : public Plugin {
    public:
        PluginMesh();

        ~PluginMesh() override = default;

        static SurfaceMesh load(const std::string &path);

        static SurfaceMesh load_obj(const std::string &path);

        static SurfaceMesh load_off(const std::string &path);

        static SurfaceMesh load_stl(const std::string &path);

        static SurfaceMesh load_ply(const std::string &path);

        static SurfaceMesh load_pmp(const std::string &path);

        static void merge_vertices(SurfaceMesh &mesh, float tol);

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render_gui(SurfaceMesh &mesh);

        void render() override;
    };
}

#endif //ENGINE24_PLUGINMESH_H