//
// Created by alex on 02.06.25.
//

#ifndef ENGINE24_COMMANDSMESH_H
#define ENGINE24_COMMANDSMESH_H

#include "Command.h"
#include "SurfaceMesh.h"

namespace Bcg::Commands{
    template<>
    struct Load<SurfaceMesh> : public AbstractCommand{
        explicit Load(entt::entity entity_id, std::string filepath) : AbstractCommand("Load<SurfaceMesh"),
                                                                      entity_id(entity_id), filepath(std::move(filepath)) {}

        void execute() const override;

        entt::entity entity_id;
        std::string filepath;
    };

    template<>
    struct Setup<SurfaceMesh> : public AbstractCommand {
        explicit Setup(entt::entity entity_id) : AbstractCommand("Setup<SurfaceMesh> "),
                                                 entity_id(entity_id) {}

        void execute() const override;

        entt::entity entity_id;
    };

    template<>
    struct Cleanup<SurfaceMesh> : public AbstractCommand {
        explicit Cleanup(entt::entity entity_id) : AbstractCommand("Cleanup<SurfaceMesh>"),
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

#endif //ENGINE24_COMMANDSMESH_H
