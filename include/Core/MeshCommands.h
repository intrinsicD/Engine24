//
// Created by alex on 18.07.24.
//

#ifndef ENGINE24_MESHCOMMANDS_H
#define ENGINE24_MESHCOMMANDS_H

#include "Command.h"
#include "entt/fwd.hpp"
#include "Fwd.h"

namespace Bcg::Commands {
    template<>
    struct Setup<SurfaceMesh> : public AbstractCommand {
        explicit Setup(entt::entity entity_id) : AbstractCommand("SetupMesh"),
                                                     entity_id(entity_id) {}

        void execute() const override;

        entt::entity entity_id;
    };

    struct ComputeFaceNormals : public AbstractCommand {
        explicit ComputeFaceNormals(entt::entity entity_id) : AbstractCommand("ComputeFaceNormals"),
                                                              entity_id(entity_id) {}

        void execute() const override;

        entt::entity entity_id;
    };
}

#endif //ENGINE24_MESHCOMMANDS_H
