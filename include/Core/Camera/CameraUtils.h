//
// Created by alex on 6/1/25.
//

#ifndef CAMERAUTILS_H
#define CAMERAUTILS_H

#include "Camera.h"

namespace Bcg {
    ViewParams GetViewParams(const Camera &camera);

    void SetViewParams(Camera &camera, const ViewParams &v_params);

    PerspectiveParams GetPerspectiveParams(const Camera &camera);

    void SetPerspectiveParams(Camera &camera, const PerspectiveParams &p_params);

    OrthoParams GetOrthoParams(const Camera &camera);

    void SetOrthoParams(Camera &camera, const OrthoParams &o_params);

    PerspectiveParams Convert(const OrthoParams &o_params);

    OrthoParams Convert(const PerspectiveParams &p_params, float depth/* = p_params.zNear*/);
}
#endif //CAMERAUTILS_H
