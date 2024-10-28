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
    };

    struct ViewParams {
        glm::vec3 eye = glm::vec3(0.0f, 0.0f, 1.0f);
        glm::vec3 center = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    };

    struct PerspectiveParams {
        float fovy = 45.0f;
        float aspect = 1.0f;
        float zNear = 0.1f;
        float zFar = 100.0f;
        bool dirty = true;
    };

    struct OrthoParams {
        float left = -1.0f;
        float right = 1.0f;
        float bottom = -1.0f;
        float top = 1.0f;
        float zNear = 0.1f;
        float zFar = 100.0f;
        bool dirty = true;
    };

    glm::vec3 get_eye(const Camera &camera) {
        return glm::vec3(camera.view[3]);
    }

    ViewParams get_view_params(const Camera &camera) {
        glm::mat4 model = glm::inverse(camera.view);
        ViewParams v_params;
        v_params.eye = model[3];
        v_params.center = model[3] - model[2];
        v_params.up = model[1];
        return v_params;
    }

    void set_view_params(Camera &camera, const ViewParams &v_params) {
        camera.view = glm::lookAt(v_params.eye, v_params.center, v_params.up);
    }

    PerspectiveParams get_perspective_params(const Camera &camera) {
        PerspectiveParams p_params;

        // Extract near and far planes
        float m22 = camera.proj[2][2];
        float m32 = camera.proj[3][2];
        p_params.zNear = m32 / (m22 - 1.0f);
        p_params.zFar = m32 / (m22 + 1.0f);

        // Extract field of view and aspect ratio
        p_params.fovy = 2.0f * atan(1.0f / camera.proj[1][1]);
        p_params.aspect = camera.proj[1][1] / camera.proj[0][0];

        return p_params;
    }

    void set_perspective_params(Camera &camera, const PerspectiveParams &p_params) {
        camera.proj_type = Camera::ProjectionType::PERSPECTIVE;
        camera.proj = glm::perspective(p_params.fovy, p_params.aspect, p_params.zNear, p_params.zFar);
    }

    OrthoParams get_ortho_params(const Camera &camera) {
        OrthoParams o_params;

        // Extract left and right
        float m00 = camera.proj[0][0];
        float m03 = camera.proj[3][0];
        o_params.left = (m03 + 1.0f) / m00;
        o_params.right = (m03 - 1.0f) / -m00;

        // Extract bottom and top
        float m11 = camera.proj[1][1];
        float m13 = camera.proj[3][1];
        o_params.bottom = (m13 + 1.0f) / m11;
        o_params.top = (m13 - 1.0f) / -m11;

        // Extract near and far planes
        float m22 = camera.proj[2][2];
        float m23 = camera.proj[3][2];
        o_params.zNear = (m23 + 1.0f) / m22;
        o_params.zFar = (m23 - 1.0f) / -m22;

        return o_params;
    }

    void set_ortho_params(Camera &camera, const OrthoParams &o_params) {
        camera.proj_type = Camera::ProjectionType::ORTHOGRAPHIC;
        camera.proj = glm::ortho(o_params.left, o_params.right, o_params.bottom, o_params.top, o_params.zNear,
                                 o_params.zFar);
    }

    PerspectiveParams Convert(const OrthoParams &o_params) {
        PerspectiveParams p_params;
        float height = o_params.top - o_params.bottom;
        p_params.aspect = (o_params.right - o_params.left) / height;
        p_params.fovy = 2.0f * atanf(height / (2.0f * o_params.zNear));
        p_params.zNear = o_params.zNear;
        p_params.zFar = o_params.zFar;
        return p_params;
    }

    OrthoParams Convert(const PerspectiveParams &p_params, float depth/* = p_params.zNear*/){
        // Compute dimensions at specified depth
        float height = 2.0f * depth * tanf(p_params.fovy / 2.0f);
        float width = height * p_params.aspect;

        // Define orthographic parameters
        OrthoParams o_params;
        o_params.left = -width / 2.0f;
        o_params.right = width / 2.0f;
        o_params.bottom = -height / 2.0f;
        o_params.top = height / 2.0f;
        o_params.zNear = depth; //set the depth where the orhto projection should match the perspective projection
        o_params.zFar = p_params.zFar;
        return o_params;
    }

    struct CameraUniformBuffer : public UniformBuffer {

    };
}

#endif //ENGINE24_CAMERA_H
