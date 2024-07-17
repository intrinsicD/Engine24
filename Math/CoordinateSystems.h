//
// Created by alex on 17.07.24.
//

#ifndef ENGINE24_COORDINATESYSTEMS_H
#define ENGINE24_COORDINATESYSTEMS_H

#include "MatVec.h"
#include "Eigen/Geometry"

namespace Bcg {
    //Todo user these functions to avoid mistakes in the camera and picker code...

    using ScreenSpacePos = Vector<float, 2>;
    using ScreenSpacePosDpiAdjusted = Vector<float, 2>;
    using ClipSpacePos = Vector<float, 3>;
    using ViewSpacePos = Vector<float, 3>;
    using WorldSpacePos = Vector<float, 3>;
    using ObjectSpacePos = Vector<float, 3>;

    ScreenSpacePosDpiAdjusted AdjustForDPI(const ScreenSpacePos &pos, float dpi_scaling_factor) {
        return {pos.x() * dpi_scaling_factor, pos.y() * dpi_scaling_factor};
    }

    ClipSpacePos screen_to_clip(const Vector<int, 4> &viewport, const ScreenSpacePosDpiAdjusted &pos, float z) {
        float xf = ((pos.x() - viewport[0]) / static_cast<float>(viewport[2])) * 2.0f - 1.0f;
        float yf = 1.0f - ((pos.y() - viewport[1]) / static_cast<float>(viewport[3])) * 2.0f; // Invert Y-axis
        float zf = z * 2.0f - 1.0f;
        return {xf, yf, zf};
    }

    ScreenSpacePosDpiAdjusted clip_to_screen(const Vector<int, 4> &viewport, const ClipSpacePos &pos, float &z) {
        z = (pos.z() + 1.0f) / 2.0f;
        float xf = (pos.x() + 1.0f) / 2.0f * static_cast<float>(viewport[2]) + viewport[0];
        float yf = (1.0f - pos.y()) / 2.0f * static_cast<float>(viewport[3]) + viewport[1]; // Invert Y-axis back
        return {xf, yf};
    }

    Vector<float, 4> transform_homogeneous(const Matrix<float, 4, 4> &mat, const Vector<float, 4> &pos) {
        Eigen::Vector<float, 4> t_pos = mat * pos;
        return t_pos / t_pos.w();
    }

    ViewSpacePos clip_to_view(const Matrix<float, 4, 4> &proj_inv, const ClipSpacePos &pos) {
        return transform_homogeneous(proj_inv, pos.homogeneous()).head<3>();
    }

    ClipSpacePos view_to_clip(const Matrix<float, 4, 4> &proj, const ViewSpacePos &pos) {
        return transform_homogeneous(proj, pos.homogeneous()).head<3>();
    }

    ViewSpacePos world_to_view(const Matrix<float, 4, 4> &view, const WorldSpacePos &pos) {
        return transform_homogeneous(view, pos.homogeneous()).head<3>();
    }

    WorldSpacePos view_to_world(const Matrix<float, 4, 4> &view_inv, const ViewSpacePos &pos) {
        return transform_homogeneous(view_inv, pos.homogeneous()).head<3>();
    }

    WorldSpacePos object_to_world(const Matrix<float, 4, 4> &model, const ObjectSpacePos &pos) {
        return transform_homogeneous(model, pos.homogeneous()).head<3>();
    }

    ObjectSpacePos world_to_object(const Matrix<float, 4, 4> &model_inv, const WorldSpacePos &pos) {
        return transform_homogeneous(model_inv, pos.homogeneous()).head<3>();
    }

}

#endif //ENGINE24_COORDINATESYSTEMS_H
