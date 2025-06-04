//
// Created by alex on 04.06.25.
//

#ifndef ENGINE24_COMMANDSSPHEREVIEW_H
#define ENGINE24_COMMANDSSPHEREVIEW_H

#include "Command.h"
#include "SphereView.h"

namespace Bcg::Commands{
    template<>
    struct Setup<SphereView> : public AbstractCommand {
        explicit Setup(entt::entity entity_id) : AbstractCommand("SetupSphereView"), entity_id(entity_id) {

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

    struct SetUniformRadiusSphereView : public AbstractCommand {
        explicit SetUniformRadiusSphereView(entt::entity entity_id, float uniform_radius) : AbstractCommand(
                "SetUniformRadiusSphereView"), entity_id(entity_id), uniform_radius(uniform_radius) {

        }

        void execute() const override;

        entt::entity entity_id;
        float uniform_radius = 1.0f; // Default radius value
    };

    struct SetColorSphereView : public AbstractCommand {
        explicit SetColorSphereView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                "SetColorSphereView"), entity_id(entity_id), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string property_name;
    };

    struct SetUniformColorSphereView : public AbstractCommand {
        explicit SetUniformColorSphereView(entt::entity entity_id, const Vector<float, 3> &uniform_color) : AbstractCommand(
                "SetUniformColorSphereView"), entity_id(entity_id), uniform_color(uniform_color) {

        }

        void execute() const override;

        entt::entity entity_id;
        Vector<float, 3> uniform_color; // Default radius value
    };

    struct SetScalarfieldSphereView : public AbstractCommand {
        explicit SetScalarfieldSphereView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                "SetScalarfieldMeshView"), entity_id(entity_id), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string property_name;
    };

    struct SetNormalSphereView : public AbstractCommand {
        explicit SetNormalSphereView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                "SetNormalSphereView"), entity_id(entity_id), property_name(property_name) {

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

#endif //ENGINE24_COMMANDSSPHEREVIEW_H
