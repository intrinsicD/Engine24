//
// Created by alex on 26.07.24.
//

#include "CameraCommands.h"
#include "Engine.h"
#include "Camera.h"
#include "AABBPool.h"

namespace Bcg{
    void CenterCamera::execute() const {
        auto &camera = Engine::Context().get<Camera>();
        if (Engine::valid(entity_id)) {
            auto &aabb = *Engine::State().get<AABBHandle>(entity_id);
            float d = aabb.diagonal().maxCoeff() / tan(camera.p_params.fovy / 2.0);
            Vector<float, 3> front = camera.v_params.front();
            camera.v_params.center = aabb.center();
            camera.v_params.eye = camera.v_params.center - front * d;
            camera.v_params.dirty = true;
        }
    }
}