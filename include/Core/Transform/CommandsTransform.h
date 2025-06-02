//
// Created by alex on 02.06.25.
//

#ifndef ENGINE24_COMMANDSTRANSFORM_H
#define ENGINE24_COMMANDSTRANSFORM_H

#include "Command.h"
#include "Transform.h"

namespace Bcg::Commands{
    template<>
    struct Setup<Transform> : public AbstractCommand {
        explicit Setup(entt::entity entity_id) : AbstractCommand("Setup<Transform>"), entity_id(entity_id) {

        }

        void execute() const override;

        entt::entity entity_id;
    };

    template<>
    struct Cleanup<Transform> : public AbstractCommand {
        explicit Cleanup(entt::entity entity_id) : AbstractCommand("Cleanup<Transform>"), entity_id(entity_id) {

        }

        void execute() const override;

        entt::entity entity_id;
    };

    struct SetIdentityTransform : public AbstractCommand {
        explicit SetIdentityTransform(entt::entity entity_id) : AbstractCommand("SetIdentityTransform"), entity_id(entity_id) {

        }

        void execute() const override;

        entt::entity entity_id;
    };
}

#endif //ENGINE24_COMMANDSTRANSFORM_H
