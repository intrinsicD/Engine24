//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_CAMERA_H
#define ENGINE24_CAMERA_H

#include "MatVec.h"

namespace Bcg {
    struct Camera {
        struct PerspParameters {
            float fovy = 45.0f;
            float aspect = 1.0f;
            float zNear = 0.1f;
            float zFar = 100.0f;
            bool dirty = false;
        } p_params;
        struct OrthoParameters {
            float left = -1.0f;
            float right = 1.0f;
            float bottom = -1.0f;
            float top = 1.0f;
            float zNear = -1.0f;
            float zFar = 1.0f;
            bool dirty = false;
        } o_params;
        struct ViewParameters {
            Vector<float, 3> eye = {0.0f, 0.0f, 1.0f};
            Vector<float, 3> center = {0.0f, 0.0f, 0.0f};
            Vector<float, 3> up = {0.0f, 1.0f, 0.0f};
            bool dirty = false;
        } v_params;
        enum class ProjectionType {
            PERSPECTIVE, ORTHOGRAPHIC
        } proj_type = ProjectionType::PERSPECTIVE;
        Matrix<float, 4, 4> view;
        Matrix<float, 4, 4> proj;
        bool dirty = false;
    };

//! OpenGL viewport matrix with parameters left, bottom, width, height
    template<typename Scalar>
    Matrix<Scalar, 4, 4> viewport_matrix(Scalar l, Scalar b, Scalar w, Scalar h) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());

        m(0, 0) = 0.5 * w;
        m(0, 3) = 0.5 * w + l;
        m(1, 1) = 0.5 * h;
        m(1, 3) = 0.5 * h + b;
        m(2, 2) = 0.5;
        m(2, 3) = 0.5;
        m(3, 3) = 1.0f;

        return m;
    }

//! inverse of OpenGL viewport matrix with parameters left, bottom, width, height
//! \sa viewport_matrix
    template<typename Scalar>
    Matrix<Scalar, 4, 4> inverse_viewport_matrix(Scalar l, Scalar b, Scalar w, Scalar h) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());

        m(0, 0) = 2.0 / w;
        m(0, 3) = -1.0 - (l + l) / w;
        m(1, 1) = 2.0 / h;
        m(1, 3) = -1.0 - (b + b) / h;
        m(2, 2) = 2.0;
        m(2, 3) = -1.0;
        m(3, 3) = 1.0f;

        return m;
    }

//! OpenGL frustum matrix with parameters left, right, bottom, top, near, far
    template<typename Scalar>
    Matrix<Scalar, 4, 4> frustum_matrix(Scalar l, Scalar r, Scalar b, Scalar t, Scalar n,
                                        Scalar f) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());

        m(0, 0) = (n + n) / (r - l);
        m(0, 2) = (r + l) / (r - l);
        m(1, 1) = (n + n) / (t - b);
        m(1, 2) = (t + b) / (t - b);
        m(2, 2) = -(f + n) / (f - n);
        m(2, 3) = -f * (n + n) / (f - n);
        m(3, 2) = -1.0f;

        return m;
    }

//! inverse of OpenGL frustum matrix with parameters left, right, bottom, top, near, far
//! \sa frustum_matrix
    template<typename Scalar>
    Matrix<Scalar, 4, 4> inverse_frustum_matrix(Scalar l, Scalar r, Scalar b, Scalar t,
                                                Scalar n, Scalar f) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());

        const Scalar nn = n + n;

        m(0, 0) = (r - l) / nn;
        m(0, 3) = (r + l) / nn;
        m(1, 1) = (t - b) / nn;
        m(1, 3) = (t + b) / nn;
        m(2, 3) = -1.0;
        m(3, 2) = (n - f) / (nn * f);
        m(3, 3) = (n + f) / (nn * f);

        return m;
    }

//! OpenGL perspective matrix with parameters field of view in y-direction,
//! aspect ratio, and distance of near and far planes
    template<typename Scalar>
    Matrix<Scalar, 4, 4> perspective_matrix(Scalar fovy, Scalar aspect, Scalar zNear,
                                            Scalar zFar) {
        Scalar t = Scalar(zNear) * tan(fovy * Scalar(std::numbers::pi / 360.0));
        Scalar b = -t;
        Scalar l = b * aspect;
        Scalar r = t * aspect;

        return frustum_matrix(l, r, b, t, Scalar(zNear), Scalar(zFar));
    }

//! inverse of perspective matrix
//! \sa perspective_matrix
    template<typename Scalar>
    Matrix<Scalar, 4, 4> inverse_perspective_matrix(Scalar fovy, Scalar aspect,
                                                    Scalar zNear, Scalar zFar) {
        Scalar t = zNear * tan(fovy * Scalar(std::numbers::pi / 360.0));
        Scalar b = -t;
        Scalar l = b * aspect;
        Scalar r = t * aspect;

        return inverse_frustum_matrix(l, r, b, t, zNear, zFar);
    }

//! OpenGL orthogonal projection matrix with parameters left, right, bottom,
//! top, near, far
    template<typename Scalar>
    Matrix<Scalar, 4, 4> ortho_matrix(Scalar left, Scalar right, Scalar bottom, Scalar top,
                                      Scalar zNear = -1, Scalar zFar = 1) {
        Matrix<Scalar, 4, 4> m(Matrix<Scalar, 4, 4>::Zero());

        m(0, 0) = Scalar(2) / (right - left);
        m(1, 1) = Scalar(2) / (top - bottom);
        m(2, 2) = -Scalar(2) / (zFar - zNear);
        m(0, 3) = -(right + left) / (right - left);
        m(1, 3) = -(top + bottom) / (top - bottom);
        m(2, 3) = -(zFar + zNear) / (zFar - zNear);
        m(3, 3) = Scalar(1);

        return m;
    }

//! OpenGL look-at camera matrix with parameters eye position, scene center, up-direction
    template<typename Scalar>
    Matrix<Scalar, 4, 4> look_at_matrix(const Vector<Scalar, 3> &eye,
                                        const Vector<Scalar, 3> &center,
                                        const Vector<Scalar, 3> &up) {
        Vector<Scalar, 3> z = (eye - center).normalized();
        Vector<Scalar, 3> x = (cross(up, z)).normalized();
        Vector<Scalar, 3> y = (cross(z, x)).normalized();

        // clang-format off
        Matrix<Scalar, 4, 4> m;
        m(0, 0) = x[0];
        m(0, 1) = x[1];
        m(0, 2) = x[2];
        m(0, 3) = -dot(x, eye);
        m(1, 0) = y[0];
        m(1, 1) = y[1];
        m(1, 2) = y[2];
        m(1, 3) = -dot(y, eye);
        m(2, 0) = z[0];
        m(2, 1) = z[1];
        m(2, 2) = z[2];
        m(2, 3) = -dot(z, eye);
        m(3, 0) = 0.0;
        m(3, 1) = 0.0;
        m(3, 2) = 0.0;
        m(3, 3) = 1.0;
        // clang-format on

        return m;
    }
}

#endif //ENGINE24_CAMERA_H
