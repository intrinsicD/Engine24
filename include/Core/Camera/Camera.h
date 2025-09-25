//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_CAMERA_H
#define ENGINE24_CAMERA_H

#include "Buffer.h"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Bcg {
    struct ViewParams {
        glm::vec3 eye = glm::vec3(0.0f, 0.0f, 1.0f);
        glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    };

    struct PerspectiveParams {
        float fovy_degrees = 45.0f;
        float aspect = 1.0f;
        float zNear = 0.1f;
        float zFar = 100.0f;
    };

    struct OrthoParams {
        float left = -1.0f;
        float right = 1.0f;
        float bottom = -1.0f;
        float top = 1.0f;
        float zNear = 0.1f;
        float zFar = 100.0f;
    };

    struct Camera {
        enum class ProjectionType {
            PERSPECTIVE, ORTHOGRAPHIC
        } proj_type = ProjectionType::PERSPECTIVE;

        // Matrices used by the renderer
        glm::mat4 view;
        glm::mat4 proj;
        bool dirty_view = false;
        bool dirty_proj = false;

        // Persisted parameters (source of truth for interactions)
        ViewParams view_params{};
        PerspectiveParams perspective_params{};
        OrthoParams ortho_params{};
    };

    struct CameraUniformBuffer : public UniformBuffer {

    };
}

#endif //ENGINE24_CAMERA_H
