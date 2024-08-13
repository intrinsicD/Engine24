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

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };

    namespace Commands{
        template<>
        struct Load<SurfaceMesh> : public AbstractCommand{
            explicit Load(entt::entity entity_id, std::string filepath) : AbstractCommand("Load<SurfaceMesh"),
                                                                          entity_id(entity_id), filepath(std::move(filepath)) {}

            void execute() const override;

            entt::entity entity_id;
            std::string filepath;
        };

        template<>
        struct Setup<SurfaceMesh> : public AbstractCommand {
            explicit Setup(entt::entity entity_id) : AbstractCommand("Setup<SurfaceMesh> "),
                                                     entity_id(entity_id) {}

            void execute() const override;

            entt::entity entity_id;
        };

        struct ComputeFaceNormals : public AbstractCommand {
            explicit ComputeFaceNormals(entt::entity entity_id) : AbstractCommand("ComputeFaceNormals"),
                                                                  entity_id(entity_id) {}

            void execute() const override;

            entt::entity entity_id;
        };
    }
}

#endif //ENGINE24_PLUGINSURFACEMESH_H
