//
// Created by alex on 30.07.24.
//

#ifndef ENGINE24_POINTCLOUDCOMMANDS_H
#define ENGINE24_POINTCLOUDCOMMANDS_H

#include "Command.h"
#include "entt/fwd.hpp"

namespace Bcg::Commands::Points {
    struct SetupPointCloud : public AbstractCommand {
        explicit SetupPointCloud(entt::entity entity_id) : AbstractCommand("SetupPointCloud"),
                                                           entity_id(entity_id) {}

        void execute() const override;

        entt::entity entity_id;
    };
}
#endif //ENGINE24_POINTCLOUDCOMMANDS_H
