//
// Created by alex on 19.07.24.
//

#include "CoordinateSystems.h"

namespace Bcg {

    PointTransformer::PointTransformer(float dpi, const Vector<int, 4> &viewport_dpi_adjusted,
                                       const Matrix<float, 4, 4> &proj,
                                       const Matrix<float, 4, 4> &view,
                                       const Matrix<float, 4, 4> &model)
            : dpi(dpi), viewport_dpi_adjusted(viewport_dpi_adjusted), proj(proj), view(view), model(model) {

    }

    Points PointTransformer::apply(const ScreenSpacePos &p, float z) const {
        Points pp;
        pp.ssp = p;
        pp.sspda = AdjustForDPI(p, dpi);
        pp.ndc = screen_to_ndc(viewport_dpi_adjusted, pp.sspda, z);
        pp.vsp = ndc_to_view(glm::inverse(proj), pp.ndc);
        pp.wsp = view_to_world(glm::inverse(view), pp.vsp);
        pp.osp = world_to_object(glm::inverse(model), pp.wsp);
        return pp;
    }

    Points PointTransformer::apply(const ScreenSpacePosDpiAdjusted &p, float z) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        pp.sspda = p;
        pp.ndc = screen_to_ndc(viewport_dpi_adjusted, pp.sspda, z);
        pp.vsp = ndc_to_view(glm::inverse(proj), pp.ndc);
        pp.wsp = view_to_world(glm::inverse(view), pp.vsp);
        pp.osp = world_to_object(glm::inverse(model), pp.wsp);

        return pp;
    }

    Points PointTransformer::apply(const NdcSpacePos &p) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        float zout;
        pp.sspda = ndc_to_screen(viewport_dpi_adjusted, pp.ndc, zout);
        pp.ndc = p;
        pp.vsp = ndc_to_view(glm::inverse(proj), pp.ndc);
        pp.wsp = view_to_world(glm::inverse(view), pp.vsp);
        pp.osp = world_to_object(glm::inverse(model), pp.wsp);
        return pp;
    }

    Points PointTransformer::apply(const ViewSpacePos &p) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        float zout;
        pp.sspda = ndc_to_screen(viewport_dpi_adjusted, pp.ndc, zout);
        pp.ndc = view_to_ndc(proj, pp.vsp);
        pp.vsp = p;
        pp.wsp = view_to_world(glm::inverse(view), pp.vsp);
        pp.osp = world_to_object(glm::inverse(model), pp.wsp);
        return pp;
    }

    Points PointTransformer::apply(const WorldSpacePos &p) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        float zout;
        pp.sspda = ndc_to_screen(viewport_dpi_adjusted, pp.ndc, zout);
        pp.ndc = view_to_ndc(proj, pp.vsp);
        pp.vsp = world_to_view(view, pp.wsp);
        pp.wsp = p;
        pp.osp = world_to_object(glm::inverse(model), pp.wsp);
        return pp;
    }

    Points PointTransformer::apply(const ObjectSpacePos &p) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        float zout;
        pp.sspda = ndc_to_screen(viewport_dpi_adjusted, pp.ndc, zout);
        pp.ndc = view_to_ndc(proj, pp.vsp);
        pp.vsp = world_to_view(view, pp.wsp);
        pp.wsp = object_to_world(model, pp.osp);
        pp.osp = p;
        return pp;
    }

    ScreenSpacePos DeadjustForDPI(const ScreenSpacePosDpiAdjusted &pos, float dpi_scaling_factor) {
        return {pos.x / dpi_scaling_factor, pos.y / dpi_scaling_factor};
    }

    ScreenSpacePosDpiAdjusted AdjustForDPI(const ScreenSpacePos &pos, float dpi_scaling_factor) {
        return {pos.x * dpi_scaling_factor, pos.y * dpi_scaling_factor};
    }

    NdcSpacePos screen_to_ndc(const Vector<int, 4> &viewport_dpi_adjusted, const ScreenSpacePosDpiAdjusted &pos, float z) {
        float xf = ((pos.x - viewport_dpi_adjusted[0]) / static_cast<float>(viewport_dpi_adjusted[2])) * 2.0f - 1.0f;
        float yf = 1.0f - ((pos.y - viewport_dpi_adjusted[1]) / static_cast<float>(viewport_dpi_adjusted[3])) * 2.0f; // Invert Y-axis
        float zf = z * 2.0f - 1.0f;
        return {xf, yf, zf};
    }

    ScreenSpacePosDpiAdjusted ndc_to_screen(const Vector<int, 4> &viewport_dpi_adjusted, const NdcSpacePos &pos, float &z_out) {
        z_out = (pos.z + 1.0f) / 2.0f;
        float xf = (pos.x + 1.0f) / 2.0f * static_cast<float>(viewport_dpi_adjusted[2]) + viewport_dpi_adjusted[0];
        float yf = (1.0f - pos.y) / 2.0f * static_cast<float>(viewport_dpi_adjusted[3]) + viewport_dpi_adjusted[1]; // Invert Y-axis back
        return {xf, yf};
    }

    inline Vector<float, 4> transform_homogeneous(const Matrix<float, 4, 4> &mat, const Vector<float, 4> &pos) {
        Vector<float, 4> t_pos = mat * pos;
        return t_pos / t_pos.w;
    }

    ViewSpacePos ndc_to_view(const Matrix<float, 4, 4> &proj_inv, const NdcSpacePos &pos) {
        return transform_homogeneous(proj_inv, Vector<float, 4>(pos, 1.0f));
    }

    NdcSpacePos view_to_ndc(const Matrix<float, 4, 4> &proj, const ViewSpacePos &pos) {
        return transform_homogeneous(proj, Vector<float, 4>(pos,  1.0f));
    }

    ViewSpacePos world_to_view(const Matrix<float, 4, 4> &view, const WorldSpacePos &pos) {
        return transform_homogeneous(view, Vector<float, 4>(pos,  1.0f));
    }

    WorldSpacePos view_to_world(const Matrix<float, 4, 4> &view_inv, const ViewSpacePos &pos) {
        return transform_homogeneous(view_inv, Vector<float, 4>(pos,  1.0f));
    }

    WorldSpacePos object_to_world(const Matrix<float, 4, 4> &model, const ObjectSpacePos &pos) {
        return transform_homogeneous(model, Vector<float, 4>(pos,  1.0f));
    }

    ObjectSpacePos world_to_object(const Matrix<float, 4, 4> &model_inv, const WorldSpacePos &pos) {
        return transform_homogeneous(model_inv, Vector<float, 4>(pos,  1.0f));
    }
}