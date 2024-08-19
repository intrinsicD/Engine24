//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_MAT4_CUH
#define ENGINE24_MAT4_CUH

#include "vec4.cuh"
#include "mat3.cuh"

namespace Bcg::cuda {
    struct mat3;

    struct mat4 {
        vec4 col0, col1, col2, col3;

        __device__ __host__ inline mat4();

        __device__ __host__ inline mat4(vec4 col0, vec4 col1, vec4 col2, vec4 col3);

        __device__ __host__ inline mat4(const mat3 &upper_left);

        __device__ __host__ inline operator mat3() const;

        __device__ __host__ inline static mat4 identity();

        __device__ __host__ inline static mat4 constant(float c);

        __device__ __host__ inline static mat4 reflect_x();

        __device__ __host__ inline static mat4 reflect_y();

        __device__ __host__ inline static mat4 reflect_z();

        __device__ __host__ inline static mat4 reflect_w();

        __device__ __host__ inline static mat4 rot(const vec3 axis, float angle);

        __device__ __host__ inline static mat4 translation(const vec3 t);

        __device__ __host__ inline static mat4 scale(const vec3 s);

        __device__ __host__ inline static mat4 shear_x(float s);

        __device__ __host__ inline static mat4 shear_y(float s);

        __device__ __host__ inline static mat4 shear_z(float s);

        __device__ __host__ inline static mat4 shear_w(float s);

        __device__ __host__ inline mat3 left_upper() const;

        __device__ __host__ inline mat3 right_upper() const;

        __device__ __host__ inline mat3 left_lower() const;

        __device__ __host__ inline mat3 right_lower() const;

        __device__ __host__ inline mat4 operator-() const;

        __device__ __host__ inline vec4 operator[](int i) const;

        __device__ __host__ inline vec4 &operator[](int i);

        __device__ __host__ inline const float &operator()(int r, int c) const;

        __device__ __host__ inline float &operator()(int r, int c);

        __device__ __host__ inline mat4 operator+(const mat4 &b) const;

        __device__ __host__ inline mat4 operator-(const mat4 &b) const;

        __device__ __host__ inline mat4 operator*(const mat4 &b) const;

        __device__ __host__ inline mat4 operator+(float b) const;

        __device__ __host__ inline mat4 operator-(float b) const;

        __device__ __host__ inline mat4 operator*(float b) const;

        __device__ __host__ inline mat4 operator/(float b) const;

        __device__ __host__ inline vec4 operator*(const vec4 &v) const;

        __device__ __host__ inline mat4 transpose() const;

        __device__ __host__ inline float determinant() const;

        __device__ __host__ inline mat4 inverse() const;

        __device__ __host__ inline mat4 adjoint() const;

        __device__ __host__ inline mat4 cofactor() const;
    };

    __device__ __host__ inline mat4::mat4() : col0(), col1(), col2(), col3() {

    }

    __device__ __host__ inline mat4::mat4(vec4 col0, vec4 col1, vec4 col2, vec4 col3) : col0(col0), col1(col1),
                                                                                        col2(col2),
                                                                                        col3(col3) {

    }

    __device__ __host__ inline mat4::mat4(const mat3 &u) : col0({u.col0.x, u.col0.y, u.col0.z, 0}),
                                                           col1({u.col1.x, u.col1.y, u.col1.z, 0}),
                                                           col2({u.col2.x, u.col2.y, u.col2.z, 0}),
                                                           col3({0, 0, 0, 1}) {

    }

    __device__ __host__ inline mat4::operator mat3() const {
        return left_upper();
    }

    __device__ __host__ inline mat4 mat4::identity() {
        return {{1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}};
    }

    __device__ __host__ inline mat4 mat4::constant(float c) {
        return {{c, c, c, c},
                {c, c, c, c},
                {c, c, c, c},
                {c, c, c, c}};
    }

    __device__ __host__ inline mat4 mat4::reflect_x() {
        return {{-1, 0, 0, 0},
                {0,  1, 0, 0},
                {0,  0, 1, 0},
                {0,  0, 0, 1}};
    }

    __device__ __host__ inline mat4 mat4::reflect_y() {
        return {{1, 0,  0, 0},
                {0, -1, 0, 0},
                {0, 0,  1, 0},
                {0, 0,  0, 1}};
    }

    __device__ __host__ inline mat4 mat4::reflect_z() {
        return {{1, 0, 0,  0},
                {0, 1, 0,  0},
                {0, 0, -1, 0},
                {0, 0, 0,  1}};
    }

    __device__ __host__ inline mat4 mat4::reflect_w() {
        return {{1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, -1}};
    }

    __device__ __host__ inline mat4 mat4::rot(const vec3 axis, float angle) {
        return mat3::rot(axis, angle);
    }

    __device__ __host__ inline mat4 mat4::translation(const vec3 t) {
        return {{1, 0, 0, t.x},
                {0, 1, 0, t.y},
                {0, 0, 1, t.z},
                {0, 0, 0, 1}};
    }

    __device__ __host__ inline mat4 mat4::scale(const vec3 s) {
        return mat3::scale(s);
    }

    __device__ __host__ inline mat4 mat4::shear_x(float s) {
        return {{1, 0, 0, 0},
                {s, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}};
    }

    __device__ __host__ inline mat4 mat4::shear_y(float s) {
        return {{1, s, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}};
    }

    __device__ __host__ inline mat4 mat4::shear_z(float s) {
        return {{1, 0, s, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}};
    }

    __device__ __host__ inline mat4 mat4::shear_w(float s) {
        return {{1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, s, 1}};
    }

    __device__ __host__ inline mat3 mat4::left_upper() const {
        return {{col0.x, col0.y, col0.z},
                {col1.x, col1.y, col1.z},
                {col2.x, col2.y, col2.z}};
    }

    __device__ __host__ inline mat3 mat4::right_upper() const {
        return {{col1.x, col1.y, col1.z},
                {col2.x, col2.y, col2.z},
                {col3.x, col3.y, col3.z}};
    }

    __device__ __host__ inline mat3 mat4::left_lower() const {
        return {{col0.y, col0.z, col0.w},
                {col1.y, col1.z, col1.w},
                {col2.y, col2.z, col2.w}};
    }

    __device__ __host__ inline mat3 mat4::right_lower() const {
        return {{col1.y, col1.z, col1.w},
                {col2.y, col2.z, col2.w},
                {col3.y, col3.z, col3.w}};
    }

    __device__ __host__ inline mat4 mat4::operator-() const {
        return {-col0, -col1, -col2, -col3};
    }

    __device__ __host__ inline vec4 mat4::operator[](int i) const {
        return *(&col0 + i);
    }

    __device__ __host__ inline vec4 &mat4::operator[](int i) {
        return *(&col0 + i);
    }

    __device__ __host__ inline const float &mat4::operator()(int r, int c) const {
        return (*this)[c][r];
    }

    __device__ __host__ inline float &mat4::operator()(int r, int c) {
        return (*this)[c][r];
    }

    __device__ __host__ inline mat4 mat4::operator+(const mat4 &b) const {
        return {col0 + b.col0, col1 + b.col1, col2 + b.col2, col3 + b.col3};
    }

    __device__ __host__ inline mat4 mat4::operator-(const mat4 &b) const {
        return {col0 - b.col0, col1 - b.col1, col2 - b.col2, col3 - b.col3};
    }

    __device__ __host__ inline mat4 mat4::operator*(const mat4 &b) const {
        return {col0 * b.col0.x + col1 * b.col0.y + col2 * b.col0.z + col3 * b.col0.w,
                col0 * b.col1.x + col1 * b.col1.y + col2 * b.col1.z + col3 * b.col1.w,
                col0 * b.col2.x + col1 * b.col2.y + col2 * b.col2.z + col3 * b.col2.w,
                col0 * b.col3.x + col1 * b.col3.y + col2 * b.col3.z + col3 * b.col3.w};
    }

    __device__ __host__ inline mat4 mat4::operator+(float b) const {
        return {col0 + b, col1 + b, col2 + b, col3 + b};
    }

    __device__ __host__ inline mat4 mat4::operator-(float b) const {
        return {col0 - b, col1 - b, col2 - b, col3 - b};
    }

    __device__ __host__ inline mat4 mat4::operator*(float b) const {
        return {col0 * b, col1 * b, col2 * b, col3 * b};
    }

    __device__ __host__ inline mat4 mat4::operator/(float b) const {
        return {col0 / b, col1 / b, col2 / b, col3 / b};
    }

    __device__ __host__ inline vec4 mat4::operator*(const vec4 &v) const {
        return {col0 * v.x + col1 * v.y + col2 * v.z + col3 * v.w};
    }

    __device__ __host__ inline mat4 mat4::transpose() const {
        return {{col0.x, col1.x, col2.x, col3.x},
                {col0.y, col1.y, col2.y, col3.y},
                {col0.z, col1.z, col2.z, col3.z},
                {col0.w, col1.w, col2.w, col3.w}};
    }

    __device__ __host__ inline double mat4_determinant(double a00, double a01, double a02, double a03,
                                                       double a10, double a11, double a12, double a13,
                                                       double a20, double a21, double a22, double a23,
                                                       double a30, double a31, double a32, double a33) {
        return a00 * mat3_determinant(a11, a12, a13,
                                      a21, a22, a23,
                                      a31, a32, a33)
               - a01 * mat3_determinant(a10, a12, a13,
                                        a20, a22, a23,
                                        a30, a32, a33)
               + a02 * mat3_determinant(a10, a11, a13,
                                        a20, a21, a23,
                                        a30, a31, a33)
               - a03 * mat3_determinant(a10, a11, a12,
                                        a20, a21, a22,
                                        a30, a31, a32);
    }

    __device__ __host__ inline float mat4::determinant() const {
        return mat4_determinant(col0.x, col1.x, col2.x, col3.x,
                                col0.y, col1.y, col2.y, col3.y,
                                col0.z, col1.z, col2.z, col3.z,
                                col0.w, col1.w, col2.w, col3.w);
    }

    __device__ __host__ inline mat4 mat4::inverse() const {
        return transpose() / determinant();
    }

    __device__ __host__ inline mat4 mat4::adjoint() const {
        return cofactor().transpose();
    }

    __device__ __host__ inline mat4 mat4::cofactor() const {
        return {//TODO check if this is correct
                // First row of cofactors
                vec4(
                        mat3_determinant(col1.y, col1.z, col1.w, col2.y, col2.z, col2.w, col3.y, col3.z, col3.w),
                        -mat3_determinant(col1.x, col1.z, col1.w, col2.x, col2.z, col2.w, col3.x, col3.z, col3.w),
                        mat3_determinant(col1.x, col1.y, col1.w, col2.x, col2.y, col2.w, col3.x, col3.y, col3.w),
                        -mat3_determinant(col1.x, col1.y, col1.z, col2.x, col2.y, col2.z, col3.x, col3.y, col3.z)
                ),
                // Second row of cofactors
                vec4(
                        -mat3_determinant(col0.y, col0.z, col0.w, col2.y, col2.z, col2.w, col3.y, col3.z, col3.w),
                        mat3_determinant(col0.x, col0.z, col0.w, col2.x, col2.z, col2.w, col3.x, col3.z, col3.w),
                        -mat3_determinant(col0.x, col0.y, col0.w, col2.x, col2.y, col2.w, col3.x, col3.y, col3.w),
                        mat3_determinant(col0.x, col0.y, col0.z, col2.x, col2.y, col2.z, col3.x, col3.y, col3.z)
                ),
                // Third row of cofactors
                vec4(
                        mat3_determinant(col0.y, col0.z, col0.w, col1.y, col1.z, col1.w, col3.y, col3.z, col3.w),
                        -mat3_determinant(col0.x, col0.z, col0.w, col1.x, col1.z, col1.w, col3.x, col3.z, col3.w),
                        mat3_determinant(col0.x, col0.y, col0.w, col1.x, col1.y, col1.w, col3.x, col3.y, col3.w),
                        -mat3_determinant(col0.x, col0.y, col0.z, col1.x, col1.y, col1.z, col3.x, col3.y, col3.z)
                ),
                // Fourth row of cofactors
                vec4(
                        -mat3_determinant(col0.y, col0.z, col0.w, col1.y, col1.z, col1.w, col2.y, col2.z, col2.w),
                        mat3_determinant(col0.x, col0.z, col0.w, col1.x, col1.z, col1.w, col2.x, col2.z, col2.w),
                        -mat3_determinant(col0.x, col0.y, col0.w, col1.x, col1.y, col1.w, col2.x, col2.y, col2.w),
                        mat3_determinant(col0.x, col0.y, col0.z, col1.x, col1.y, col1.z, col2.x, col2.y, col2.z)
                )
        };
    }

    __device__ __host__ inline mat4 operator+(float a, const mat4 &b) {
        return {a + b.col0, a + b.col1, a + b.col2, a + b.col3};
    }

    __device__ __host__ inline mat4 operator-(float a, const mat4 &b) {
        return {a - b.col0, a - b.col1, a - b.col2, a - b.col3};
    }

    __device__ __host__ inline mat4 operator*(float a, const mat4 &b) {
        return {a * b.col0, a * b.col1, a * b.col2, a * b.col3};
    }

    __device__ __host__ inline mat4 operator/(float a, const mat4 &b) {
        return {a / b.col0, a / b.col1, a / b.col2, a / b.col3};
    }


}

#endif //ENGINE24_MAT4_CUH
