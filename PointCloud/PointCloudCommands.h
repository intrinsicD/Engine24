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

    struct ComputePointCloudLocalPcasKnn : public AbstractCommand {
        explicit ComputePointCloudLocalPcasKnn(entt::entity entity_id, int num_closest) : AbstractCommand(
                "ComputePointCloudLocalPcasKnn"),
                                                                                          entity_id(entity_id),
                                                                                          num_closest(num_closest) {}

        void execute() const override;

        entt::entity entity_id;
        int num_closest;
    };
}
#endif //ENGINE24_POINTCLOUDCOMMANDS_H
