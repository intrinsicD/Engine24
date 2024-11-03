//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_CAMERA_H
#define ENGINE24_CAMERA_H

#include "Buffer.h"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Bcg {
    struct Camera {
        enum class ProjectionType {
            PERSPECTIVE, ORTHOGRAPHIC
        } proj_type = ProjectionType::PERSPECTIVE;

        glm::mat4 view;
        glm::mat4 proj;
        bool dirty_view = false;
        bool dirty_proj = false;
    };

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

    ViewParams GetViewParams(const Camera &camera);

    void SetViewParams(Camera &camera, const ViewParams &v_params);

    PerspectiveParams GetPerspectiveParams(const Camera &camera);

    void SetPerspectiveParams(Camera &camera, const PerspectiveParams &p_params);

    OrthoParams GetOrthoParams(const Camera &camera);

    void SetOrthoParams(Camera &camera, const OrthoParams &o_params);

    PerspectiveParams Convert(const OrthoParams &o_params);

    OrthoParams Convert(const PerspectiveParams &p_params, float depth/* = p_params.zNear*/);

    struct CameraUniformBuffer : public UniformBuffer {

    };
}

#endif //ENGINE24_CAMERA_H
