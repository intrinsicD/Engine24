//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_CAMERACOMMANDS_H
#define ENGINE24_CAMERACOMMANDS_H

#include "Command.h"
#include "entt/fwd.hpp"

namespace Bcg {
    struct CenterCamera : public AbstractCommand {
        explicit CenterCamera(entt::entity entity_id) : AbstractCommand("CenterCamera"), entity_id(entity_id) {}

        void execute() const override;

        entt::entity entity_id;
    };
}
#endif //ENGINE24_CAMERACOMMANDS_H
