//
// Created by alex on 02.06.25.
//

#include "CommandsCamera.h"
#include "ModuleCamera.h"

namespace Bcg::Commands {
    void CenterCameraAtDistance::execute() const {
        ModuleCamera::center_camera_at_distance(center, distance);
    }

    void FitNearAndFarToDistance::execute() const {
        ModuleCamera::fit_near_and_far_to_distance(distance);
    }
}