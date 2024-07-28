//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_RIGIDTRANSFORM_H
#define ENGINE24_RIGIDTRANSFORM_H

#include "MatVec.h"
#include "Eigen/Geometry"

namespace Bcg {
    struct TransformParameters {
        Vector<float, 3> scale;
        Vector<float, 3> angle_axis;
        Vector<float, 3> position;
    };

    class RigidTransform : public Eigen::Transform<float, 3, Eigen::Affine> {
    public:
        using Base = Eigen::Transform<float, 3, Eigen::Affine>;
        using Base::operator=;

        RigidTransform() : Base() {

        }

        explicit RigidTransform(const Matrix<float, 4, 4> &matrix) : Base(matrix) {

        }

        explicit RigidTransform(const Matrix<float, 3, 3> &matrix) : Base(Identity()) {
            Vector<float, 3> scale_vector(matrix.colwise().norm());
            linear() = Eigen::AngleAxisf((matrix.array().colwise() / scale_vector.array()).matrix()).toRotationMatrix();
            scale(scale_vector);
        }

        explicit RigidTransform(const TransformParameters &params) : Base(Identity()) {
            float angle = params.angle_axis.norm();
            linear() = Eigen::AngleAxisf(angle, params.angle_axis / angle).toRotationMatrix();
            scale(params.scale);
            SetPosition(params.position);
        }

        [[nodiscard]] TransformParameters Decompose() const {
            TransformParameters params;
            params.scale = linear().colwise().norm();
            Eigen::AngleAxisf rot((linear().array().colwise() / params.scale.array()).matrix());
            params.angle_axis = rot.angle() * rot.axis();
            params.position = Position();
            return std::move(params);
        }

        [[nodiscard]] Vector<float, 3> Right() const {
            return m_matrix.col(0).head<3>();
        }

        [[nodiscard]] Vector<float, 3> Up() const {
            return m_matrix.col(1).head<3>();
        }

        [[nodiscard]] Vector<float, 3> Dir() const {
            return m_matrix.col(2).head<3>();
        }

        [[nodiscard]] Vector<float, 3> Position() const {
            return m_matrix.col(3).head<3>();
        }

        RigidTransform &SetRight(const Vector<float, 3> &v) {
            m_matrix.col(0) = v.homogeneous();
            return *this;
        }

        RigidTransform &SetUp(const Vector<float, 3> &v) {
            m_matrix.col(1) = v.homogeneous();
            return *this;
        }

        RigidTransform &SetDir(const Vector<float, 3> &v) {
            m_matrix.col(2) = v.homogeneous();
            return *this;
        }

        RigidTransform &SetPosition(const Vector<float, 3> &v) {
            m_matrix.col(3) = v.homogeneous();
            return *this;
        }

        static RigidTransform Identity() {
            RigidTransform t;
            t.setIdentity();
            return t;
        }

        static RigidTransform Translation(const Vector<float, 3> &vt) {
            auto t = RigidTransform();
            t.SetPosition(vt);
            return t;
        }

        static RigidTransform Scale(const Vector<float, 3> &vt) {
            auto t = RigidTransform();
            t.linear() = vt.asDiagonal();
            return t;
        }

        static RigidTransform Rotation(const Vector<float, 3> &angle_axis) {
            auto t = RigidTransform();
            t.linear() = Eigen::AngleAxisf(angle_axis.norm(), angle_axis.normalized()).toRotationMatrix();
            return t;
        }
    };
}

#endif //ENGINE24_RIGIDTRANSFORM_H
