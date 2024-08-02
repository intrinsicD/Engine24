//
// Created by alex on 02.08.24.
//

#ifndef ENGINE24_SPHERECOMMANDS_H
#define ENGINE24_SPHERECOMMANDS_H

#include "Command.h"
#include "entt/fwd.hpp"

namespace Bcg::Commands::View {
    struct SetupSphereView : public AbstractCommand {
        explicit SetupSphereView(entt::entity entity_id) : AbstractCommand("SetupSphereView"), entity_id(entity_id) {

        }

        void execute() const override;

        entt::entity entity_id;
    };

    struct SetPositionSphereView : public AbstractCommand {
        explicit SetPositionSphereView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                "SetPositionSphereView"), entity_id(entity_id), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string property_name;
    };

    struct SetRadiusSphereView : public AbstractCommand {
        explicit SetRadiusSphereView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                "SetRadiusSphereView"), entity_id(entity_id), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string property_name;
    };

    struct SetColorSphereView : public AbstractCommand {
        explicit SetColorSphereView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                "SetColorSphereView"), entity_id(entity_id), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string property_name;
    };

    struct SetIndicesSphereView : public AbstractCommand {
        explicit SetIndicesSphereView(entt::entity entity_id, std::vector<unsigned int> &indices) : AbstractCommand(
                "SetIndicesSphereView"), entity_id(entity_id), indices(indices) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::vector<unsigned int> &indices;
    };
}

#endif //ENGINE24_SPHERECOMMANDS_H
