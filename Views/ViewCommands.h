//
// Created by alex on 31.07.24.
//

#ifndef ENGINE24_VIEWCOMMANDS_H
#define ENGINE24_VIEWCOMMANDS_H

#include "Command.h"
#include "entt/fwd.hpp"

namespace Bcg::Commands::View {
    struct SetupPointsView : public AbstractCommand {
        explicit SetupPointsView(entt::entity entity_id) : AbstractCommand("SetupPointsView"), entity_id(entity_id) {

        }

        void execute() const override;

        entt::entity entity_id;
    };

    struct SetupGraphView : public AbstractCommand {
        SetupGraphView(entt::entity entity_id) : AbstractCommand("SetupGraphView"), entity_id(entity_id) {

        }

        void execute() const override;

        entt::entity entity_id;
    };
}

#endif //ENGINE24_VIEWCOMMANDS_H
