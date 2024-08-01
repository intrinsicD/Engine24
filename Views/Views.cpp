//
// Created by alex on 18.07.24.
//
#include "Views.h"
#include "Logger.h"
#include "glad/gl.h"

namespace Bcg {
    Vector<float, 4> PickingView::encode(entt::entity entity_id) {
        uint32_t id = static_cast<uint32_t>(entity_id);

        // Normalize each component to the range [0, 1]
        float r = ((id >> 24) & 0xFF) / 255.0f;
        float g = ((id >> 16) & 0xFF) / 255.0f;
        float b = ((id >> 8) & 0xFF) / 255.0f;
        float a = (id & 0xFF) / 255.0f;

        return {r, g, b, a};
    }

    entt::entity PickingView::encode(const Vector<float, 4> &picking_color) {
        // Convert back from normalized [0, 1] to 8-bit [0, 255]
        uint32_t r = static_cast<uint32_t>(picking_color.x() * 255.0f);
        uint32_t g = static_cast<uint32_t>(picking_color.y() * 255.0f);
        uint32_t b = static_cast<uint32_t>(picking_color.z() * 255.0f);
        uint32_t a = static_cast<uint32_t>(picking_color.w() * 255.0f);

        // Combine the components back into a single integer
        uint32_t id = (r << 24) | (g << 16) | (b << 8) | a;

        return static_cast<entt::entity>(id);
    }

    void PickingView::draw() {
        //Todo implement!
        Log::TODO("Picking is not yet implemented here!");
    }

    void PointView::draw() {
        glDrawElements(GL_POINTS, num_indices, GL_UNSIGNED_INT, nullptr);
    }

    void TriangleView::draw() {
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, nullptr);
    }
}