//
// Created by alex on 6/1/25.
//

#ifndef AABBCOMMANDS_H
#define AABBCOMMANDS_H

#include "AABB.h"
#include "Command.h"
#include "entt/fwd.hpp"

namespace Bcg::Commands {
    template<>
    struct Setup<AABB<float>> : public AbstractCommand {
        explicit Setup(entt::entity entity_id) : AbstractCommand("Setup<AABB>"), entity_id(entity_id) {

        }

        void execute() const override;

        entt::entity entity_id;
    };

    template<>
    struct Cleanup<AABB<float>> : public AbstractCommand {
        explicit Cleanup(entt::entity entity_id) : AbstractCommand("Cleanup<AABB>"), entity_id(entity_id) {

        }

        void execute() const override;

        entt::entity entity_id;
    };

    struct CenterAndScaleByAABB : public AbstractCommand {
        explicit CenterAndScaleByAABB(entt::entity entity_id, std::string property_name) : AbstractCommand(
                "CenterAndScaleByAABB"), entity_id(entity_id), property_name(property_name) {

        }

        void execute() const override;

        entt::entity entity_id;
        std::string property_name;
    };
}

#endif //AABBCOMMANDS_H
