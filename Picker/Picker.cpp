//
// Created by alex on 03.07.24.
//

#include "Picker.h"
#include "Engine.h"
#include "imgui.h"
#include "../Camera/Camera.h"
#include "Graphics.h"
#include "Mouse.h"
#include "EventsCallbacks.h"
#include "Eigen/Geometry"

namespace Bcg {
    Vector<float, 3> screen_to_ndc(const Vector<int, 4> &viewport, float x, float y, float z) {
        float xf = ((float) x - (float) viewport[0]) / ((float) viewport[2]) * 2.0f - 1.0f;
        float yf = ((float) y - (float) viewport[1]) / ((float) viewport[3]) * 2.0f - 1.0f;
        float zf = z * 2.0f - 1.0f;
        return {xf, yf, zf};
    }
}