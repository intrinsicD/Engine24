//
// Created by alex on 19.07.24.
//

#include "CoordinateSystems.h"
#include "Eigen/Geometry"

namespace Bcg {

    PointTransformer::PointTransformer(float dpi, const Eigen::Vector<int, 4> &viewport_dpi_adjusted,
                                       const Eigen::Matrix<float, 4, 4> &proj,
                                       const Eigen::Matrix<float, 4, 4> &view,
                                       const Eigen::Matrix<float, 4, 4> &model)
            : dpi(dpi), viewport_dpi_adjusted(viewport_dpi_adjusted), proj(proj), view(view), model(model) {

    }

    Points PointTransformer::apply(const ScreenSpacePos &p, float z) const {
        Points pp;
        pp.ssp = p;
        pp.sspda = AdjustForDPI(p, dpi);
        pp.ndc = ScreenToNdc(viewport_dpi_adjusted, pp.sspda, z);

        pp.vsp = NdcToView(proj.inverse(), pp.ndc);
        pp.wsp = ViewToWorld(view.inverse(), pp.vsp);
        pp.osp = WorldToObject(model.inverse(), pp.wsp);
        return pp;
    }

    Points PointTransformer::apply(const ScreenSpacePosDpiAdjusted &p, float z) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        pp.sspda = p;
        pp.ndc = ScreenToNdc(viewport_dpi_adjusted, pp.sspda, z);

        pp.vsp = NdcToView(proj.inverse(), pp.ndc);
        pp.wsp = ViewToWorld(view.inverse(), pp.vsp);
        pp.osp = WorldToObject(model.inverse(), pp.wsp);
        return pp;
    }

    Points PointTransformer::apply(const NdcSpacePos &p) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        float zout;
        pp.sspda = NdcToScreen(viewport_dpi_adjusted, pp.ndc, zout);
        pp.ndc = p;

        pp.vsp = NdcToView(proj.inverse(), pp.ndc);
        pp.wsp = ViewToWorld(view.inverse(), pp.vsp);
        pp.osp = WorldToObject(model.inverse(), pp.wsp);
        return pp;
    }

    Points PointTransformer::apply(const ViewSpacePos &p) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        float zout;
        pp.sspda = NdcToScreen(viewport_dpi_adjusted, pp.ndc, zout);
        pp.ndc = ViewToNdc(proj, pp.vsp);
        pp.vsp = p;
        pp.wsp = ViewToWorld(view.inverse(), pp.vsp);
        pp.osp = WorldToObject(model.inverse(), pp.wsp);
        return pp;
    }

    Points PointTransformer::apply(const WorldSpacePos &p) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        float zout;
        pp.sspda = NdcToScreen(viewport_dpi_adjusted, pp.ndc, zout);
        pp.ndc = ViewToNdc(proj, pp.vsp);
        pp.vsp = WorldToView(view, pp.wsp);
        pp.wsp = p;
        pp.osp = WorldToObject(model.inverse(), pp.wsp);
        return pp;
    }

    Points PointTransformer::apply(const ObjectSpacePos &p) const {
        Points pp;
        pp.ssp = DeadjustForDPI(pp.sspda, dpi);
        float zout;
        pp.sspda = NdcToScreen(viewport_dpi_adjusted, pp.ndc, zout);
        pp.ndc = ViewToNdc(proj, pp.vsp);
        pp.vsp = WorldToView(view, pp.wsp);
        pp.wsp = ObjectToWorld(model, pp.osp);
        pp.osp = p;
        return pp;
    }

    ScreenSpacePos DeadjustForDPI(const ScreenSpacePosDpiAdjusted &pos, float dpi_scaling_factor) {
        return {pos.x() / dpi_scaling_factor, pos.y() / dpi_scaling_factor};
    }

    ScreenSpacePosDpiAdjusted AdjustForDPI(const ScreenSpacePos &pos, float dpi_scaling_factor) {
        return {pos.x() * dpi_scaling_factor, pos.y() * dpi_scaling_factor};
    }

    NdcSpacePos ScreenToNdc(const Eigen::Vector<int, 4> &viewport_dpi_adjusted, const ScreenSpacePosDpiAdjusted &pos, float z) {
        float xf = ((pos.x() - viewport_dpi_adjusted[0]) / static_cast<float>(viewport_dpi_adjusted[2])) * 2.0f - 1.0f;
        float yf = 1.0f - ((pos.y() - viewport_dpi_adjusted[1]) / static_cast<float>(viewport_dpi_adjusted[3])) * 2.0f; // Invert Y-axis
        float zf = z * 2.0f - 1.0f;
        return {xf, yf, zf};
    }

    ScreenSpacePosDpiAdjusted NdcToScreen(const Eigen::Vector<int, 4> &viewport_dpi_adjusted, const NdcSpacePos &pos, float &z_out) {
        z_out = (pos.z() + 1.0f) / 2.0f;
        float xf = (pos.x() + 1.0f) / 2.0f * static_cast<float>(viewport_dpi_adjusted[2]) + viewport_dpi_adjusted[0];
        float yf = (1.0f - pos.y()) / 2.0f * static_cast<float>(viewport_dpi_adjusted[3]) + viewport_dpi_adjusted[1]; // Invert Y-axis back
        return {xf, yf};
    }

    inline Eigen::Vector<float, 3> transform_homogeneous(const Eigen::Matrix<float, 4, 4> &mat, const Eigen::Vector<float, 3> &pos) {
        Eigen::Vector<float, 4> t_pos = mat * Eigen::Vector<float, 4>(pos.x(), pos.y(), pos.z(), 1.0f);
        return t_pos.head<3>() / t_pos.w();
    }

    ViewSpacePos NdcToView(const Eigen::Matrix<float, 4, 4> &proj_inv, const NdcSpacePos &pos) {
        return transform_homogeneous(proj_inv, pos);
    }

    NdcSpacePos ViewToNdc(const Eigen::Matrix<float, 4, 4> &proj, const ViewSpacePos &pos) {
        return transform_homogeneous(proj, pos);
    }

    ViewSpacePos WorldToView(const Eigen::Matrix<float, 4, 4> &view, const WorldSpacePos &pos) {
        return transform_homogeneous(view, pos);
    }

    WorldSpacePos ViewToWorld(const Eigen::Matrix<float, 4, 4> &view_inv, const ViewSpacePos &pos) {
        return transform_homogeneous(view_inv, pos);
    }

    WorldSpacePos ObjectToWorld(const Eigen::Matrix<float, 4, 4> &model, const ObjectSpacePos &pos) {
        return transform_homogeneous(model, pos);
    }

    ObjectSpacePos WorldToObject(const Eigen::Matrix<float, 4, 4> &model_inv, const WorldSpacePos &pos) {
        return transform_homogeneous(model_inv, pos);
    }
}