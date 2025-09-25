//
// Created by alex on 30.10.24.
//

#include "CameraUtils.h"

namespace Bcg{
    ViewParams GetViewParams(const Camera &camera) {
        return camera.view_params;
    }

    void SetViewParams(Camera &camera, const ViewParams &v_params) {
        camera.view_params = v_params;
        camera.view = glm::lookAt(v_params.eye, v_params.center, v_params.up);
        camera.dirty_view = true;
    }

    PerspectiveParams GetPerspectiveParams(const Camera &camera) {
        return camera.perspective_params;
    }

    void SetPerspectiveParams(Camera &camera, const PerspectiveParams &p_params) {
        camera.proj_type = Camera::ProjectionType::PERSPECTIVE;
        camera.perspective_params = p_params;
        camera.proj = glm::perspective(glm::radians(p_params.fovy_degrees), p_params.aspect, p_params.zNear, p_params.zFar);
        camera.dirty_proj = true;
    }

    OrthoParams GetOrthoParams(const Camera &camera) {
        return camera.ortho_params;
    }

    void SetOrthoParams(Camera &camera, const OrthoParams &o_params) {
        camera.proj_type = Camera::ProjectionType::ORTHOGRAPHIC;
        camera.ortho_params = o_params;
        camera.proj = glm::ortho(o_params.left, o_params.right, o_params.bottom, o_params.top, o_params.zNear,
                                 o_params.zFar);
        camera.dirty_proj = true;
    }

    PerspectiveParams Convert(const OrthoParams &o_params) {
        PerspectiveParams p_params;
        float height = o_params.top - o_params.bottom;
        float width = o_params.right - o_params.left;
        p_params.aspect = width / height;
        // fovy in degrees
        p_params.fovy_degrees = glm::degrees(2.0f * atanf(height / (2.0f * o_params.zNear)));
        p_params.zNear = o_params.zNear;
        p_params.zFar = o_params.zFar;
        return p_params;
    }

    OrthoParams Convert(const PerspectiveParams &p_params, float depth/* = p_params.zNear*/){
        // Compute dimensions at specified depth
        float height = 2.0f * depth * tanf(glm::radians(p_params.fovy_degrees) * 0.5f);
        float width = height * p_params.aspect;

        // Define orthographic parameters
        OrthoParams o_params;
        o_params.left = -width * 0.5f;
        o_params.right = width * 0.5f;
        o_params.bottom = -height * 0.5f;
        o_params.top = height * 0.5f;
        o_params.zNear = depth; // set the depth where the ortho projection should match the perspective projection
        o_params.zFar = p_params.zFar;
        return o_params;
    }
}