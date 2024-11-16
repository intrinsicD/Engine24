//
// Created by alex on 11/10/24.
//

#ifndef POINTCLOUDVERTEXCOLORS_H
#define POINTCLOUDVERTEXCOLORS_H

#include <utility>

#include "glm/vec3.hpp"

#include "Command.h"

namespace Bcg::Commands {
    struct SetPointCloudVertexColors3D : public AbstractCommand {
        SetPointCloudVertexColors3D(entt::entity entity_id, const glm::vec3 *&colors, size_t count,
                          std::string property_name = "") : AbstractCommand(
                                                          "SetPointCloudVertexColors3D"),
                                                      entity_id(entity_id), colors(colors), count(count),
                                                      property_name(std::move(property_name)) {
        }

        void execute() const override;

        entt::entity entity_id;
        const glm::vec3 *colors;
        size_t count;
        mutable std::string property_name;
    };

    struct SetPointCloudVertexColorsScalarfield : public AbstractCommand {
        SetPointCloudVertexColorsScalarfield(entt::entity entity_id, const float *&sfield, size_t count) : AbstractCommand(
                "SetPointCloudVertexColorsScalarfield"),
            entity_id(entity_id), sfield(sfield), count(count) {
        }

        void execute() const override;

        entt::entity entity_id;
        const float *sfield;
        size_t count;
    };

    struct SetPointCloudVertexColorsSelection3D : public AbstractCommand {
        SetPointCloudVertexColorsSelection3D(entt::entity entity_id, const std::uint32_t *selected_indices,
                                   const glm::vec3 *&selected_colors, size_t selected_count) : AbstractCommand(
                "SetPointCloudVertexColorsSelection3D"),
            entity_id(entity_id), selected_indices(selected_indices), selected_colors(selected_colors),
            selected_count(selected_count) {
        }

        void execute() const override;

        entt::entity entity_id;
        const std::uint32_t *selected_indices;
        const glm::vec3 *selected_colors;
        size_t selected_count;
    };

    struct SetPointCloudVertexColorsSelectionScalarfield : public AbstractCommand {
        SetPointCloudVertexColorsSelectionScalarfield(entt::entity entity_id, const std::uint32_t *selected_indices,
                                            const float *&selected_colors, size_t selected_count) : AbstractCommand(
                "SetPointCloudVertexColorsSelectionScalarfield"),
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

#endif //POINTCLOUDVERTEXCOLORS_H
