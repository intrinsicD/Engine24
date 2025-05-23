//
// Created by alex on 22.05.25.
//

#ifndef ENGINE24_CAMERAUTILS_H
#define ENGINE24_CAMERAUTILS_H

#include "Camera.h"
#include "Rotation.h"

namespace Bcg {
    //Compute the frustum of the camera
    template<typename T>
    void RotateAroundCenter(Camera<T> &camera, const AngleAxis<T> &angle_axis) {
        auto view_params = camera.get_view_params();
        Eigen::Vector<T, 3> offset = view_params.eye - view_params.center;
        // Rotate that offset, then re-translate
        view_params.eye = view_params.center + angle_axis * offset;
        // Rotate the up-vector so the camera doesn't roll
        view_params.up = angle_axis * view_params.up;
        camera.set_view_params(view_params);
    }

    template<typename T>
    void FocusPoint(Camera<float> &camera, const Eigen::Vector<float, 3> &point) {
        auto view_params = camera.get_view_params();
        view_params.center = point;
        camera.set_view_params(view_params);
    }

    template<typename T>
    typename Camera<T>::ViewParams GetViewParamsFromModelMatrix(const Eigen::Matrix<T, 4, 4> &model_matrix) {
        typename Camera<T>::ViewParams view_params;
        view_params.eye = model_matrix.col(3).template head<3>();
        Eigen::Vector<T, 3> forward = -model_matrix.col(2).template head<3>();
        view_params.center = view_params.eye + forward;
        view_params.up = model_matrix.col(1).template head<3>();
        return view_params;
    }
}

#endif //ENGINE24_CAMERAUTILS_H
