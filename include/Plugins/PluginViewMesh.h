//
// Created by alex on 04.08.24.
//

#ifndef ENGINE24_PLUGINVIEWMESH_H
#define ENGINE24_PLUGINVIEWMESH_H

#include "Plugin.h"
#include "Command.h"
#include "MeshView.h"
#include "MatVec.h"

namespace Bcg {
    class PluginViewMesh : public Plugin {
    public:
        PluginViewMesh() : Plugin("ViewMesh") {}

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
        struct Setup<MeshView> : public AbstractCommand {
            Setup(entt::entity entity_id) : AbstractCommand("Setup<MeshView>"), entity_id(entity_id) {

            }

            void execute() const override;

            entt::entity entity_id;
        };

        template<>
        struct Cleanup<MeshView> : public AbstractCommand {
            Cleanup(entt::entity entity_id) : AbstractCommand("Cleanup<MeshView>"), entity_id(entity_id) {

            }

            void execute() const override;

            entt::entity entity_id;
        };

        struct SetPositionMeshView : public AbstractCommand {
            explicit SetPositionMeshView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                    "SetPositionMeshView"), entity_id(entity_id), property_name(property_name) {

            }

            void execute() const override;

            entt::entity entity_id;
            std::string property_name;
        };

        struct SetNormalMeshView : public AbstractCommand {
            explicit SetNormalMeshView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                    "SetNormalMeshView"), entity_id(entity_id), property_name(property_name) {

            }

            void execute() const override;

            entt::entity entity_id;
            std::string property_name;
        };

        struct SetColorMeshView : public AbstractCommand {
            explicit SetColorMeshView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                    "SetColorMeshView"), entity_id(entity_id), property_name(property_name) {

            }

            void execute() const override;

            entt::entity entity_id;
            std::string property_name;
        };

        struct SetScalarfieldMeshView : public AbstractCommand {
            explicit SetScalarfieldMeshView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                    "SetScalarfieldMeshView"), entity_id(entity_id), property_name(property_name) {

            }

            void execute() const override;

            entt::entity entity_id;
            std::string property_name;
        };

        struct SetTrianglesMeshView : public AbstractCommand {
            explicit SetTrianglesMeshView(entt::entity entity_id, std::vector<Vector<unsigned int, 3>> &tris)
            : AbstractCommand(
            "SetTrianglesMeshView"), entity_id(entity_id), tris(tris) {

            }

            void execute() const override;

            entt::entity entity_id;
            std::vector<Vector<unsigned int, 3>> &tris;
        };
    }
}
#endif //ENGINE24_PLUGINVIEWMESH_H
