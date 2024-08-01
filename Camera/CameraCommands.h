//
// Created by alex on 26.07.24.
//

#ifndef ENGINE24_CAMERACOMMANDS_H
#define ENGINE24_CAMERACOMMANDS_H

#include "Command.h"
#include "MatVec.h"
#include "entt/fwd.hpp"

namespace Bcg {
    struct CenterCameraAtDistance : public AbstractCommand {
        explicit CenterCameraAtDistance(const Vector<float, 3> &center, float distance = 3) :
                AbstractCommand("CenterCamera"),
                center(center),
                distance(distance) {}

        void execute() const override;

        Vector<float, 3> center;
        float distance;
    };
}
#endif //ENGINE24_CAMERACOMMANDS_H
