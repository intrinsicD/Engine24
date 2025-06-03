//
// Created by alex on 03.06.25.
//

#ifndef ENGINE24_COMMANDSMESHVIEW_H
#define ENGINE24_COMMANDSMESHVIEW_H

#include "Command.h"
#include "MeshView.h"

namespace Bcg::Commands {

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

    struct SetUniforColorMeshView : public AbstractCommand {
        explicit SetUniforColorMeshView(entt::entity entity_id, const Vector<float, 3> &color) : AbstractCommand(
                "SetUniforColorMeshView"), entity_id(entity_id), color(color) {

        }

        void execute() const override;

        entt::entity entity_id;
        Vector<float, 3> color;
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

} // namespace Bcg::Commands

#endif //ENGINE24_COMMANDSMESHVIEW_H
