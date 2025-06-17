//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_PICKERSYSTEM_H
#define ENGINE24_PICKERSYSTEM_H

#include "entt/entity/registry.hpp"
#include "EntitySelection.h"
#include "Camera.h"
#include "FrameBuffer.h"
#include "Program.h"
#include <optional>

namespace Bcg{
    class PickerSystem {
    public:
        // The constructor should initialize the FBO and load the shader.
        PickerSystem(uint32_t viewport_width, uint32_t viewport_height);

        // Public method to request a pick at a specific mouse position for the next frame.
        void request_pick(float mouse_x, float mouse_y);

        // The main update function to be called during the rendering phase of the game loop.
        // It will only perform the expensive picking render pass if a pick was requested.
        void update(entt::registry& registry, const Camera& camera, EntitySelection& selection_state);

        // Must be called when the main window's viewport resizes.
        void on_viewport_resize(uint32_t width, uint32_t height);

    private:
        // The actual rendering logic.
        void perform_pick_render_pass(entt::registry& registry, const Camera& camera);

        Framebuffer m_fbo;
        Program m_picking_shader;

        // Store the pick request. std::optional is perfect for this.
        std::optional<glm::vec2> m_pick_request_pos;
    };
}

#endif //ENGINE24_PICKERSYSTEM_H
