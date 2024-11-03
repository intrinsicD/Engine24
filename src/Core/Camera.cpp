//
// Created by alex on 30.10.24.
//

#include "Camera.h"

namespace Bcg{
    ViewParams GetViewParams(const Camera &camera) {
        glm::mat4 model = glm::inverse(camera.view);
        ViewParams v_params;
        v_params.eye = model[3];
        v_params.center = model[3] - model[2];
        v_params.up = model[1];
        return v_params;
    }

    void SetViewParams(Camera &camera, const ViewParams &v_params) {
        camera.view = glm::lookAt(v_params.eye, v_params.center, v_params.up);
        camera.dirty_view = true;
    }

    PerspectiveParams GetPerspectiveParams(const Camera &camera) {
        PerspectiveParams p_params;

        // Extract near and far planes
        float m22 = camera.proj[2][2];
        float m32 = camera.proj[3][2];
        p_params.zNear = m32 / (m22 - 1.0f);
        p_params.zFar = m32 / (m22 + 1.0f);

        // Extract field of view and aspect ratio
        p_params.fovy_degrees = glm::degrees(2.0f * atan(1.0f / camera.proj[1][1]));
        p_params.aspect = camera.proj[1][1] / camera.proj[0][0];

        return p_params;
    }

    void SetPerspectiveParams(Camera &camera, const PerspectiveParams &p_params) {
        camera.proj_type = Camera::ProjectionType::PERSPECTIVE;
        camera.proj = glm::perspective(glm::radians(p_params.fovy_degrees), p_params.aspect, p_params.zNear, p_params.zFar);
        camera.dirty_proj = true;
    }

    OrthoParams GetOrthoParams(const Camera &camera) {
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

    void SetOrthoParams(Camera &camera, const OrthoParams &o_params) {
        camera.proj_type = Camera::ProjectionType::ORTHOGRAPHIC;
        camera.proj = glm::ortho(o_params.left, o_params.right, o_params.bottom, o_params.top, o_params.zNear,
                                 o_params.zFar);
        camera.dirty_proj = true;
    }

    PerspectiveParams Convert(const OrthoParams &o_params) {
        PerspectiveParams p_params;
        float height = o_params.top - o_params.bottom;
        p_params.aspect = (o_params.right - o_params.left) / height;
        p_params.fovy_degrees = 2.0f * atanf(height / (2.0f * o_params.zNear));
        p_params.zNear = o_params.zNear;
        p_params.zFar = o_params.zFar;
        return p_params;
    }

    OrthoParams Convert(const PerspectiveParams &p_params, float depth/* = p_params.zNear*/){
        // Compute dimensions at specified depth
        float height = 2.0f * depth * tanf(p_params.fovy_degrees / 2.0f);
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
}