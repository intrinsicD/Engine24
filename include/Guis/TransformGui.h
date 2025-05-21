//
// Created by alex on 30.10.24.
//

#ifndef ENGINE24_TRANSFORMGUI_H
#define ENGINE24_TRANSFORMGUI_H

#include "Transform.h"

namespace Bcg::Gui {
    bool Show(Transform<float> &transform);

    bool Show(Transform<float>::Parameters &t_params);

    void Show(const Eigen::Matrix<float, 4, 4> &mat);

    bool ShowGuizmo(Eigen::Matrix<float, 4, 4> &mat, Eigen::Matrix<float, 4, 4> &delta, bool &is_scaling);
}

#endif //ENGINE24_TRANSFORMGUI_H
