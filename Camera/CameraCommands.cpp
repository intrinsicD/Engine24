//
// Created by alex on 26.07.24.
//

#include "CameraCommands.h"
#include "Engine.h"
#include "Camera.h"

namespace Bcg {
    void CenterCameraAtDistance::execute() const {
        auto &camera = Engine::Context().get<Camera>();

        Vector<float, 3> front = camera.v_params.front();
        camera.v_params.center = center;
        camera.v_params.eye = camera.v_params.center - front * distance / tan(camera.p_params.fovy / 2.0);
        camera.v_params.dirty = true;

        FitNearAndFarToDistance(distance).execute();
    }

    void FitNearAndFarToDistance::execute() const {
        auto &camera = Engine::Context().get<Camera>();

        if (camera.proj_type == Camera::ProjectionType::PERSPECTIVE) {
            camera.p_params.zNear = distance / 100.0f;
            camera.p_params.zFar = distance * 3.0f;
            camera.p_params.dirty = true;
        } else if (camera.proj_type == Camera::ProjectionType::ORTHOGRAPHIC) {
            camera.o_params.zNear = distance / 100.0f;
            camera.o_params.zFar = distance * 3.0f;
            camera.o_params.dirty = true;
        }
    }
}