//
// Created by alex on 26.06.24.
//

#include "PluginOpenGLRenderer.h"
#include "Core/Engine.h"
#include "Camera/Camera.h"
#include "glad/gl.h"

namespace Bcg {
    static unsigned int uboCamera = -1;

    PluginOpenGLRenderer::PluginOpenGLRenderer() : Plugin("OpenGLRenderer") {
        if (uboCamera == -1) {
            glGenBuffers(1, &uboCamera);
            glBindBuffer(GL_UNIFORM_BUFFER, uboCamera);
            glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(Matrix<float, 4, 4>), NULL, GL_STATIC_DRAW);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
            glBindBufferBase(GL_UNIFORM_BUFFER, 0, uboCamera);
        }
    }

    void linkUniformBlockToUBO(unsigned int program, const char *blockName,
                               unsigned int bindingPoint) {
        GLuint blockIndex = glGetUniformBlockIndex(program, blockName);
        if (blockIndex != GL_INVALID_INDEX) {
            glUniformBlockBinding(program, blockIndex, bindingPoint);
        }
    }

    struct Hide {

    };

    struct Show{

    };

    void PluginOpenGLRenderer::unhide_entity(entt::entity entity_id) {
        Engine::State().remove<Hide>(entity_id);
        Engine::State().emplace<Show>(entity_id);
    }

    void PluginOpenGLRenderer::hide_entity(entt::entity entity_id) {
        Engine::State().remove<Show>(entity_id);
        Engine::State().emplace_or_replace<Hide>(entity_id);
    }

    void PluginOpenGLRenderer::activate() {
        Plugin::activate();
    }

    void PluginOpenGLRenderer::begin_frame() {}

    void PluginOpenGLRenderer::update() {}

    void PluginOpenGLRenderer::end_frame() {}

    void PluginOpenGLRenderer::deactivate() {
        Plugin::deactivate();
    }

    void PluginOpenGLRenderer::render_menu() {}

    void PluginOpenGLRenderer::render_gui() {}

    void PluginOpenGLRenderer::render() {
        auto *camera = Engine::Context().find<Camera>();
        if (!camera) {
            return;
        }
        //determine visibility of all visual entities (Frustum Culling)
        //figure out on which primitives to work and if a hierarchy is even better
        std::vector<entt::entity> visibleMeshes;
        std::vector<entt::entity> visibleGraphs;
        std::vector<entt::entity> visiblePointClouds;
/*
        for (auto &entity: Engine::State().view<Show>()) {
            if (isEntityVisible(camera, entity)) {
                switch (entity.type()) {
                    case EntityType::Mesh:
                        visibleMeshes.push_back(static_cast<Mesh *>(&entity));
                        break;
                    case EntityType::Graph:
                        visibleGraphs.push_back(static_cast<Graph *>(&entity));
                        break;
                    case EntityType::PointCloud:
                        visiblePointClouds.push_back(static_cast<PointCloud *>(&entity));
                        break;
                    default:
                        break;
                }
            }
        }

        // Render all visible meshes
        for (auto entity_id: visibleMeshes) {
            renderMesh(entity_id);
        }

        // Render all visible graphs
        for (auto entity_id: visibleGraphs) {
            renderGraph(entity_id);
        }

        // Render all visible point clouds
        for (auto entity_id: visiblePointClouds) {
            renderPointCloud(entity_id);
        }*/
    }
}