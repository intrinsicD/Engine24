//
// Created by alex on 17.06.25.
//
#include "PickerSystem.h"
#include "MeshView.h" // Or whatever holds your mesh
#include "WorldTransformComponent.h"
#include "ModuleGraphics.h"
#include "Engine.h"
#include "Picker.h"

#include "glad/gl.h" // Or your OpenGL header

namespace Bcg {

    PickerSystem::PickerSystem(uint32_t viewport_width, uint32_t viewport_height)
            : m_fbo(viewport_width, viewport_height),
              m_picking_shader("../Shaders/glsl/picking_vs.glsl", "../Shaders/glsl/picking_fs.glsl") // Assuming your Shader class loads from file
    {}

    void PickerSystem::request_pick(float mouse_x, float mouse_y) {
        // Store the coordinates. The actual picking will happen in the Update method.
        m_pick_request_pos = {mouse_x, mouse_y};
    }

    void PickerSystem::on_viewport_resize(uint32_t width, uint32_t height) {
        m_fbo.resize(width, height);
    }

    void PickerSystem::update(entt::registry &registry, const Camera &camera) {
        // Only run the expensive render pass if a pick was actually requested this frame.
        if (!m_pick_request_pos.has_value()) {
            return;
        }

        // --- 1. Perform the render pass ---
        perform_pick_render_pass(registry, camera);

        // --- 2. Read the pixel and update selection ---
        glm::vec2 pick_pos = m_pick_request_pos.value();

        // The framebuffer's Y coordinate (0 at bottom) is often inverted relative
        // to window/mouse coordinates (0 at top). We must flip it.
        uint32_t flipped_y = m_fbo.get_height() - static_cast<uint32_t>(pick_pos.y);

        int entity_id = m_fbo.read_pixel(static_cast<uint32_t>(pick_pos.x), flipped_y);
        auto &picked = registry.ctx().get<Picked>();
        if (entity_id != -1) {
            picked.entity.id = static_cast<entt::entity>(entity_id);
        } else {
            picked.entity = {};
        }

        // --- 3. Clear the request so we don't pick again next frame ---
        m_pick_request_pos.reset();
    }


    void PickerSystem::perform_pick_render_pass(entt::registry &registry, const Bcg::Camera &camera) {
        // --- A. Bind resources for the picking pass ---
        m_fbo.bind();
        m_picking_shader.bind();

        // --- B. Clear the framebuffer ---
        // We use glClearBufferiv for integer framebuffers.
        // The clear value -1 represents "no entity".
        int clear_value = -1;
        glClearBufferiv(GL_COLOR, 0, &clear_value);
        // We still need to clear the depth buffer for correct occlusion.
        glClear(GL_DEPTH_BUFFER_BIT);

        // --- C. Set camera uniforms ---
        m_picking_shader.set_uniform4fm("u_View", glm::value_ptr(camera.view));
        m_picking_shader.set_uniform4fm("u_Projection", glm::value_ptr(camera.proj));

        // --- D. Render all relevant entities ---
        // Query for entities that can be rendered (and thus, picked).
        auto rendergroup = registry.view<MeshView>();
        for (auto entity_id: rendergroup) {
            auto &view = registry.get<MeshView>(entity_id);
            if (view.hide) continue;

            view.vao.bind();
            // Set the entity ID uniform for this specific draw call.
            // We must cast the entt::entity handle to a plain integer.
            m_picking_shader.set_uniform1i("u_EntityID", static_cast<int32_t>(entity_id));

            // Set the model matrix
            if (Engine::has<WorldTransformComponent>(entity_id)) {
                auto &transform = Engine::State().get<WorldTransformComponent>(entity_id);
                m_picking_shader.set_uniform4fm("model", glm::value_ptr(transform.world_transform), false);
            } else {
                m_picking_shader.set_uniform4fm("model", glm::value_ptr(glm::mat4(1.0f)), false);
            }

            ModuleGraphics::draw_triangles(view.num_indices);
        }

        // --- E. Unbind framebuffer to return to normal rendering ---
        m_fbo.unbind();
    }
}