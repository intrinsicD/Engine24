//
// Created by alex on 02.08.24.
//

#ifndef ENGINE24_PLUGINVIEWSPHERE_H
#define ENGINE24_PLUGINVIEWSPHERE_H

#include "Plugin.h"
#include "Command.h"
#include "SphereView.h"

namespace Bcg {
    class PluginViewSphere : public Plugin {
    public:
        PluginViewSphere() : Plugin("ViewSphere") {}

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

        struct SetColorSphereView : public AbstractCommand {
            explicit SetColorSphereView(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                    "SetColorSphereView"), entity_id(entity_id), property_name(property_name) {

            }

            void execute() const override;

            entt::entity entity_id;
            std::string property_name;
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
}

#endif //ENGINE24_PLUGINVIEWSPHERE_H
