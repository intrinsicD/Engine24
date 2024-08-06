//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_SELECTIONCOMMANDS_H
#define ENGINE24_SELECTIONCOMMANDS_H

#include "Command.h"
#include "entt/fwd.hpp"

namespace Bcg::Commands {
    struct MarkPoints : public AbstractCommand {
        MarkPoints(entt::entity entity_id, const std::string &property_name) : AbstractCommand("MarkPoints"),
                                                                               entity_id(entity_id),
                                                                               property_name(property_name) {}

        void execute() const override;

        entt::entity entity_id;
        std::string property_name;
    };

    struct EnableVertexSelection : public AbstractCommand {
        EnableVertexSelection(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                "EnableVertexSelection"),
                                                                                          entity_id(entity_id) {}

        void execute() const override;

        entt::entity entity_id;
    };
}

#endif //ENGINE24_SELECTIONCOMMANDS_H
