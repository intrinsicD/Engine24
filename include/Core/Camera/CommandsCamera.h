//
// Created by alex on 02.06.25.
//

#ifndef ENGINE24_COMMANDSCAMERA_H
#define ENGINE24_COMMANDSCAMERA_H

#include "Command.h"
#include "MatVec.h"

namespace Bcg::Commands{
    struct CenterCameraAtDistance : public AbstractCommand {
        explicit CenterCameraAtDistance(const Vector<float, 3> &center, float distance = 3) :
                AbstractCommand("CenterCamera"),
                center(center),
                distance(distance) {}

        void execute() const override;

        Vector<float, 3> center;
        float distance;
    };

    struct FitNearAndFarToDistance : public AbstractCommand {
        explicit FitNearAndFarToDistance(float distance = 3) : AbstractCommand("FitNearAndFarToDistance"),
                                                               distance(distance) {

        }

        void execute() const override;

        float distance;
    };
}

#endif //ENGINE24_COMMANDSCAMERA_H
