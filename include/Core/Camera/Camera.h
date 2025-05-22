//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_CAMERA_H
#define ENGINE24_CAMERA_H

#include <utility>

#include "Buffer.h"
#include "MatUtils.h"
#include "Eigen/Geometry"

namespace Bcg {
    template<typename T>
    class Camera {
    public:
        struct ViewParams {
            Eigen::Vector<T, 3> eye = Eigen::Vector<T, 3>(0.0f, 0.0f, 1.0f);
            Eigen::Vector<T, 3> center = Eigen::Vector<T, 3>(0.0f, 0.0f, 0.0f);
            Eigen::Vector<T, 3> up = Eigen::Vector<T, 3>(0.0f, 1.0f, 0.0f);
        };

        struct PerspectiveParams {
            T fovy_degrees = 45.0f;
            T aspect = 1.0f;
            T zNear = 0.1f;
            T zFar = 100.0f;
        };

        struct OrthoParams {
            T left = -1.0f;
            T right = 1.0f;
            T bottom = -1.0f;
            T top = 1.0f;
            T zNear = 0.1f;
            T zFar = 100.0f;
        };

        Camera() = default;

        Camera(ViewParams view_params,
               PerspectiveParams perspective_params)
                : view_params(std::move(view_params)),
                  perspective_params(std::move(perspective_params)) {
            proj_type = ProjectionType::PERSPECTIVE;
        }

        Camera(ViewParams view_params,
               OrthoParams ortho_params)
                : view_params(std::move(view_params)),
                  ortho_params(std::move(ortho_params)) {
            proj_type = ProjectionType::PERSPECTIVE;
        }

        enum class ProjectionType {
            PERSPECTIVE, ORTHOGRAPHIC
        };

        void set_view_params(const ViewParams &v_params) {
            view_params = v_params;
            dirty_view = true;
        }

        ViewParams &get_view_params() {
            return view_params;
        }

        void set_perspective_params(const PerspectiveParams &p_params) {
            perspective_params = p_params;
            proj_type = ProjectionType::PERSPECTIVE;
            dirty_proj = true;
        }

        PerspectiveParams &get_perspective_params() {
            return perspective_params;
        }

        void set_ortho_params(const OrthoParams &o_params) {
            ortho_params = o_params;
            proj_type = ProjectionType::ORTHOGRAPHIC;
            dirty_proj = true;
        }

        OrthoParams &get_ortho_params() {
            return ortho_params;
        }

        ProjectionType get_projection_type() const {
            return proj_type;
        }

        const Eigen::Matrix<T, 4, 4> &get_view() const{
            if (dirty_view) {
                view = lookAt(view_params.eye, view_params.center, view_params.up);
                dirty_view = false;
            }
            return view;
        }

        const Eigen::Matrix<T, 4, 4> &get_proj() const {
            if (dirty_proj) {
                if (proj_type == ProjectionType::PERSPECTIVE) {
                    proj = perspective(perspective_params.fovy_degrees,
                                       perspective_params.aspect,
                                       perspective_params.zNear,
                                       perspective_params.zFar);
                } else {
                    proj = ortho(ortho_params.left, ortho_params.right,
                                 ortho_params.bottom, ortho_params.top,
                                 ortho_params.zNear, ortho_params.zFar);
                }
                dirty_proj = false;
            }
            return proj;
        }

        Eigen::Matrix<T, 4, 4> get_model_matrix() {
            return get_view().inverse();
        }


    private:
        static Eigen::Matrix<T, 4, 4> lookAt(const Eigen::Vector<T, 3> &eye,
                                             const Eigen::Vector<T, 3> &center,
                                             const Eigen::Vector<T, 3> &up) {
            // Also known as the "view" matrix.
            // It transforms world coordinates to camera (view/eye) coordinates.

            // 1. Calculate the camera's local coordinate system axes.
            // z_axis (forward, but points from target to eye, or -direction camera is looking)
            // We want the direction vector the camera is looking AT.
            Eigen::Vector<T, 3> f = (center - eye).normalized(); // Forward vector (camera's -Z in OpenGL convention)

            // x_axis (right)
            Eigen::Vector<T, 3> s = f.cross(up).normalized(); // Side/Right vector (camera's +X)
            // If f and up are parallel (e.g. looking straight up/down), s will be zero.
            // A robust implementation would handle this, e.g., by picking a default side vector.
            // For simplicity, we assume valid inputs.

            // y_axis (up)
            Eigen::Vector<T, 3> u = s.cross(f); // Recompute up-vector to be orthogonal (camera's +Y)
            // It should already be normalized if s and f are unit and orthogonal.

            Eigen::Matrix<T, 4, 4> Result = Eigen::Matrix<T, 4, 4>::Identity();

            // The rotation part of the view matrix is the inverse of the camera's orientation matrix.
            // The camera's orientation matrix would have s, u, -f as its columns.
            // Its inverse has s, u, -f as its rows.
            Result(0, 0) = s.x();
            Result(0, 1) = s.y();
            Result(0, 2) = s.z();
            Result(1, 0) = u.x();
            Result(1, 1) = u.y();
            Result(1, 2) = u.z();
            Result(2, 0) = -f.x();
            Result(2, 1) = -f.y();
            Result(2, 2) = -f.z(); // -f is the direction camera's Z axis points

            // The translation part moves the world origin to the camera's position,
            // then the rotation orients it.
            // Effectively, translate by -eye: T_view = R_view * T_(-eye)
            // So the last column is R_view * (-eye)
            // -s.dot(eye)
            // -u.dot(eye)
            //  f.dot(eye)  (since the third row of R_view is -f)
            Result(0, 3) = -s.dot(eye);
            Result(1, 3) = -u.dot(eye);
            Result(2, 3) = f.dot(eye); //  This is -(-f.dot(eye))

            return Result;
        }

        static Eigen::Matrix<T, 4, 4> perspective(T fovy_degrees, T aspect, T zNear, T zFar) {
            // Creates a perspective projection matrix.
            // Maps view space coordinates to clip space.
            // fovy_degrees: Field of view in the Y direction, in degrees.
            // aspect: Aspect ratio (width / height).
            // zNear: Distance to the near clipping plane.
            // zFar: Distance to the far clipping plane.

            // Convert fovy to radians
            T fovy_radians = DegreesToRadians(fovy_degrees);

            // Ensure zNear and zFar are positive and zFar > zNear
            // Add error handling or assertions if needed.
            // assert(zNear > 0.0f && zFar > 0.0f && zFar > zNear);
            // assert(aspect > 0.0f);
            // assert(fovy_degrees > 0.0f && fovy_degrees < 180.0f);

            T tanHalfFovy = std::tan(fovy_radians / 2.0f);

            Eigen::Matrix<T, 4, 4> Result = Eigen::Matrix<T, 4, 4>::Zero(); // Initialize with zeros

            // OpenGL perspective projection matrix (column-major order in docs,
            // but Eigen is row-major by default, so we fill M(row, col))
            //
            //   | (1/(aspect*tanHF))     0            0              0            |
            //   |       0            (1/tanHF)        0              0            |
            //   |       0                0    -(f+n)/(f-n)   -(2*f*n)/(f-n)   |
            //   |       0                0           -1              0            |
            //
            // where tanHF = tan(fovy_radians / 2)
            // f = zFar, n = zNear

            Result(0, 0) = 1.0f / (aspect * tanHalfFovy);
            Result(1, 1) = 1.0f / tanHalfFovy;
            Result(2, 2) = -(zFar + zNear) / (zFar - zNear);
            Result(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);
            Result(3, 2) = -1.0f;
            // Result(3,3) is 0, which is correct.

            return Result;
        }

        static Eigen::Matrix<T, 4, 4> ortho(T left, T right, T bottom, T top, T zNear, T zFar) {
            // Creates an orthographic projection matrix.
            // Maps view space coordinates (within the defined box) to clip space [-1, 1] cube.

            // Ensure valid ranges
            // assert(right > left);
            // assert(top > bottom);
            // assert(zFar > zNear); // For typical OpenGL (zNear and zFar can be negative if they represent distances along -Z)
            // If zNear and zFar are *distances from the camera*, they should be positive.
            // Let's assume they are coordinates along the camera's Z axis, where -Z is forward.
            // If so, zNear would be typically -0.1 and zFar -100.0.
            // However, the common GLM/OpenGL ortho takes positive distances from the camera.
            // The formula below assumes zNear and zFar are distances,
            // and it maps zNear to -1 and zFar to +1 in NDC.
            // Or, if they are coordinates along the Z axis, then zNear could be greater than zFar
            // (e.g. near plane at z=-1, far plane at z=-100).
            // The formula from learnopengl.com (and GLM) uses:
            // P22 = -2 / (far - near)
            // P23 = -(far + near) / (far - near)
            // This maps the near plane to -1 and the far plane to 1 in NDC.

            Eigen::Matrix<T, 4, 4> Result = Eigen::Matrix<T, 4, 4>::Identity(); // Start with identity

            // Scale factors
            Result(0, 0) = 2.0f / (right - left);
            Result(1, 1) = 2.0f / (top - bottom);
            Result(2, 2) = -2.0f / (zFar - zNear); // Maps [zNear, zFar] to [-1, 1] if z is increasing
            // If zNear/zFar are distances from eye (positive), and camera looks along -Z,
            // then view space z is negative.
            // This formula works correctly for mapping view-space z to NDC z.

            // Translation factors (to shift the center to origin before scaling)
            Result(0, 3) = -(right + left) / (right - left);
            Result(1, 3) = -(top + bottom) / (top - bottom);
            Result(2, 3) = -(zFar + zNear) / (zFar - zNear);
            // Result(3,3) is 1 from Identity()

            return Result;
        }

        ViewParams view_params{};
        PerspectiveParams perspective_params{};
        OrthoParams ortho_params{};
        ProjectionType proj_type = ProjectionType::PERSPECTIVE;

        mutable Eigen::Matrix<T, 4, 4> view;
        mutable Eigen::Matrix<T, 4, 4> proj;

        mutable bool dirty_view = true;
        mutable bool dirty_proj = true;
    };

    struct CameraUniformBuffer : public UniformBuffer {

    };
}

#endif //ENGINE24_CAMERA_H
