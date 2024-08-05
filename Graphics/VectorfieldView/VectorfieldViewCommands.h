//
// Created by alex on 05.08.24.
//

#ifndef ENGINE24_VECTORFIELDVIEWCOMMANDS_H
#define ENGINE24_VECTORFIELDVIEWCOMMANDS_H


#include "Command.h"
#include "entt/fwd.hpp"

namespace Bcg::Commands::View {
    struct SetupVectorfieldView : public AbstractCommand {
        explicit SetupVectorfieldView(entt::entity entity_id, const std::string &vectorfield_name) : AbstractCommand(
                "SetupVectorfieldView"), entity_id(entity_id), vectorfield_name(vectorfield_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string vectorfield_name;
    };

    struct SetPositionVectorfieldView : public AbstractCommand {
        explicit SetPositionVectorfieldView(entt::entity entity_id, const std::string &vectorfield_name, const std::string &property_name) : AbstractCommand(
                "SetPositionVectorfieldView"), entity_id(entity_id), vectorfield_name(vectorfield_name), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string vectorfield_name;
        std::string property_name;
    };

    struct SetLengthVectorfieldView : public AbstractCommand {
        explicit SetLengthVectorfieldView(entt::entity entity_id, const std::string &vectorfield_name, const std::string &property_name) : AbstractCommand(
                "SetLengthVectorfieldView"), entity_id(entity_id), vectorfield_name(vectorfield_name), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string vectorfield_name;
        std::string property_name;
    };

    struct SetColorVectorfieldView : public AbstractCommand {
        explicit SetColorVectorfieldView(entt::entity entity_id, const std::string &vectorfield_name, const std::string &property_name) : AbstractCommand(
                "SetColorVectorfieldView"), entity_id(entity_id), vectorfield_name(vectorfield_name), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string vectorfield_name;
        std::string property_name;
    };

    struct SetVectorVectorfieldView : public AbstractCommand {
        explicit SetVectorVectorfieldView(entt::entity entity_id, const std::string &vectorfield_name, const std::string &property_name) : AbstractCommand(
                "SetVectorVectorfieldView"), entity_id(entity_id), vectorfield_name(vectorfield_name), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string vectorfield_name;
        std::string property_name;
    };

    struct SetIndicesVectorfieldView : public AbstractCommand {
        explicit SetIndicesVectorfieldView(entt::entity entity_id, const std::string &vectorfield_name, std::vector<unsigned int> &indices)
                : AbstractCommand(
                "SetIndicesVectorfieldView"), entity_id(entity_id), vectorfield_name(vectorfield_name),indices(indices) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string vectorfield_name;
        std::vector<unsigned int> &indices;
    };
}


#endif //ENGINE24_VECTORFIELDVIEWCOMMANDS_H
