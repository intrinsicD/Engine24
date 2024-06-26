//
// Created by alex on 26.06.24.
//

#include "PluginOpenGLRenderer.h"
#include "Engine.h"
#include "Camera.h"
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
        //upload uniform buffer objects like camera
        if (camera->dirty) {
            glBindBuffer(GL_UNIFORM_BUFFER, uboCamera);
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(camera->view), camera->view.data());
            glBufferSubData(GL_UNIFORM_BUFFER, sizeof(camera->view), sizeof(camera->proj), camera->proj.data());
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
            camera->dirty = false;
        }
        //determine visibility of all visual entities (Frustum Culling)


        //render all visible meshes
        //render all visible graphs
        //render all visible point clouds
    }
}