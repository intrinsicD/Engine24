//
// Created by alex on 29.07.24.
//

#ifndef ENGINE24_HIERARCHYCOMMANDS_H
#define ENGINE24_HIERARCHYCOMMANDS_H

#include "Command.h"
#include "entt/fwd.hpp"

namespace Bcg {
    struct UpdateTransforms : public AbstractCommand {
        explicit UpdateTransforms(entt::entity entity_id) : AbstractCommand("UpdateTransfroms"), entity_id(entity_id) {}

        void execute() const override;

        entt::entity entity_id;
    };
}

#endif //ENGINE24_HIERARCHYCOMMANDS_H
