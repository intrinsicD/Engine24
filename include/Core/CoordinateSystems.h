//
// Created by alex on 17.07.24.
//

#ifndef ENGINE24_COORDINATESYSTEMS_H
#define ENGINE24_COORDINATESYSTEMS_H

#include "MatVec.h"
#include "Eigen/Geometry"

namespace Bcg {
    //Todo use these functions to avoid mistakes in the camera and picker code...

    struct ScreenSpacePos : public Vector<float, 2> {
        using Vector<float, 2>::Vector;
    };

    struct ScreenSpacePosDpiAdjusted : public Vector<float, 2> {
        using Vector<float, 2>::Vector;
    };

    struct NdcSpacePos : public Vector<float, 3> {
        using Vector<float, 3>::Vector;
    };

    struct ViewSpacePos : public Vector<float, 3> {
        using Vector<float, 3>::Vector;
    };

    struct WorldSpacePos : public Vector<float, 3> {
        using Vector<float, 3>::Vector;
    };

    struct ObjectSpacePos : public Vector<float, 3> {
        using Vector<float, 3>::Vector;
    };

    struct Points {
        ScreenSpacePos ssp;
        ScreenSpacePosDpiAdjusted sspda;
        NdcSpacePos ndc;
        ViewSpacePos vsp;
        WorldSpacePos wsp;
        ObjectSpacePos osp;
    };

    struct PointTransformer {
        float dpi;
        const Vector<int, 4> &viewport_dpi_adjusted;
        const Matrix<float, 4, 4> &proj;
        const Matrix<float, 4, 4> &view;
        const Matrix<float, 4, 4> &model;

        PointTransformer(float dpi, const Vector<int, 4> &viewport_dpi_adjusted,
                         const Matrix<float, 4, 4> &proj,
                         const Matrix<float, 4, 4> &view,
                         const Matrix<float, 4, 4> &model = Matrix<float, 4, 4>(1.0f));

        [[nodiscard]] Points apply(const ScreenSpacePos &p, float z = 0) const;

        [[nodiscard]] Points apply(const ScreenSpacePosDpiAdjusted &p, float z = 0) const;

        [[nodiscard]] Points apply(const NdcSpacePos &p) const;

        [[nodiscard]] Points apply(const ViewSpacePos &p) const;

        [[nodiscard]] Points apply(const WorldSpacePos &p) const;

        [[nodiscard]] Points apply(const ObjectSpacePos &p) const;
    };

    ScreenSpacePos DeadjustForDPI(const ScreenSpacePosDpiAdjusted &pos, float dpi_scaling_factor);

    ScreenSpacePosDpiAdjusted AdjustForDPI(const ScreenSpacePos &pos, float dpi_scaling_factor);

    NdcSpacePos ScreenToNdc(const Vector<int, 4> &viewport, const ScreenSpacePosDpiAdjusted &pos, float z);

    ScreenSpacePosDpiAdjusted NdcToScreen(const Vector<int, 4> &viewport, const NdcSpacePos &pos, float &z_out);

    ViewSpacePos NdcToView(const Matrix<float, 4, 4> &proj_inv, const NdcSpacePos &pos);

    NdcSpacePos ViewToNdc(const Matrix<float, 4, 4> &proj, const ViewSpacePos &pos);

    ViewSpacePos WorldToView(const Matrix<float, 4, 4> &view, const WorldSpacePos &pos);

    WorldSpacePos ViewToWorld(const Matrix<float, 4, 4> &view_inv, const ViewSpacePos &pos);

    WorldSpacePos ObjectToWorld(const Matrix<float, 4, 4> &model, const ObjectSpacePos &pos);

    ObjectSpacePos WorldToObject(const Matrix<float, 4, 4> &model_inv, const WorldSpacePos &pos);
}

#endif //ENGINE24_COORDINATESYSTEMS_H
