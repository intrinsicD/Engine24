//
// Created by alex on 11/10/24.
//

#ifndef VERTEXCOLORS_H
#define VERTEXCOLORS_H

#include <utility>

#include "glm/vec3.hpp"

#include "Command.h"

namespace Bcg::Commands {
    struct SetVertexColors3D : public AbstractCommand {
        SetVertexColors3D(entt::entity entity_id, const glm::vec3 *&colors, size_t count,
                          std::string property_name = "") : AbstractCommand(
                                                          "SetVertexColors3D"),
                                                      entity_id(entity_id), colors(colors), count(count),
                                                      property_name(std::move(property_name)) {
        }

        void execute() const override;

        entt::entity entity_id;
        const glm::vec3 *colors;
        size_t count;
        mutable std::string property_name;
    };

    struct SetVertexColorsScalarfield : public AbstractCommand {
        SetVertexColorsScalarfield(entt::entity entity_id, const float *&sfield, size_t count) : AbstractCommand(
                "SetVertexColorsScalarfield"),
            entity_id(entity_id), sfield(sfield), count(count) {
        }

        void execute() const override;

        entt::entity entity_id;
        const float *sfield;
        size_t count;
    };

    struct SetVertexColorsSelection3D : public AbstractCommand {
        SetVertexColorsSelection3D(entt::entity entity_id, const std::uint32_t *selected_indices,
                                   const glm::vec3 *&selected_colors, size_t selected_count) : AbstractCommand(
                "SetVertexColorsSelection3D"),
            entity_id(entity_id), selected_indices(selected_indices), selected_colors(selected_colors),
            selected_count(selected_count) {
        }

        void execute() const override;

        entt::entity entity_id;
        const std::uint32_t *selected_indices;
        const glm::vec3 *selected_colors;
        size_t selected_count;
    };

    struct SetVertexColorsSelectionScalarfield : public AbstractCommand {
        SetVertexColorsSelectionScalarfield(entt::entity entity_id, const std::uint32_t *selected_indices,
                                            const float *&selected_colors, size_t selected_count) : AbstractCommand(
                "SetVertexColorsSelectionScalarfield"),
            entity_id(entity_id), selected_indices(selected_indices), selected_colors(selected_colors),
            selected_count(selected_count) {
        }

        void execute() const override;

        entt::entity entity_id;
        const std::uint32_t *selected_indices;
        const float *selected_colors;
        size_t selected_count;
    };
}

#endif //VERTEXCOLORS_H
