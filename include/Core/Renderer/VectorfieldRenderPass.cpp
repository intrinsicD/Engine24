//
// Created by alex on 6/17/25.
//

#include "VectorfieldRenderPass.h"
#include "OpenGLState.h"
#include "VectorfieldView.h"
#include "AssetManager.h"

namespace Bcg {
    VectorfieldRenderPass::VectorfieldRenderPass(AssetManager &asset_manager)
        : m_asset_manager(asset_manager) {
        m_static_mesh_shader = asset_manager.get_program("VectorfieldsProgram");
    }

    void VectorfieldRenderPass::execute(IRenderer *renderer, entt::registry &registry, const Camera &camera) {
        OpenGLState openGlState;
        openGlState.set_camera(camera);

        auto view = registry.view<VectorfieldViews>();
        for (auto entity_id: view) {
            auto &vectorfield_views = registry.get<VectorfieldViews>(entity_id);
            for (auto &[name, vectorfield_view]: vectorfield_views.vectorfields) {
                vectorfield_view.program = m_static_mesh_shader;
                vectorfield_view.vao.bind();
                vectorfield_view.program.use();
                vectorfield_view.program.set_uniform("u_camera", camera.get_view_projection_matrix());
                vectorfield_view.vao.draw();
            }
        }
    }
}