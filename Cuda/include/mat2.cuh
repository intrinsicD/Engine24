//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_MAT2_CUH
#define ENGINE24_MAT2_CUH

#include "vec2.cuh"

namespace Bcg::cuda {
    struct mat2 {
        vec2 col0, col1;

        __device__ __host__ inline mat2() : col0(), col1() {

        }

        __device__ __host__ inline mat2(vec2 col0, vec2 col1) : col0(col0), col1(col1) {

        }

        __device__ __host__ inline mat2(float m00, float m01,
                                        float m10, float m11) : col0(m00, m10), col1(m01, m11) {

        }

        __device__ __host__ inline static mat2 identity();

        __device__ __host__ inline static mat2 constant(float c);

        __device__ __host__ inline static mat2 reflect_x();

        __device__ __host__ inline static mat2 reflect_y();

        __device__ __host__ inline static mat2 rot(float angle);

        __device__ __host__ inline static mat2 project(float angle);

        __device__ __host__ inline static mat2 shear_x(float s);

        __device__ __host__ inline static mat2 shear_y(float s);

        __device__ __host__ inline mat2 operator-() const;

        __device__ __host__ inline vec2 operator[](int i) const;

        __device__ __host__ inline vec2 &operator[](int i);

        __device__ __host__ inline const float &operator()(int r, int c) const;

        __device__ __host__ inline float &operator()(int r, int c);

        __device__ __host__ inline mat2 operator+(const mat2 &b) const;

        __device__ __host__ inline mat2 operator-(const mat2 &b) const;

        __device__ __host__ inline mat2 operator*(const mat2 &b) const;

        __device__ __host__ inline mat2 operator+(float b) const;

        __device__ __host__ inline mat2 operator-(float b) const;

        __device__ __host__ inline mat2 operator*(float b) const;

        __device__ __host__ inline mat2 operator/(float b) const;

        __device__ __host__ inline vec2 operator*(const vec2 &v) const;

        __device__ __host__ inline mat2 transpose() const;

        __device__ __host__ inline float determinant() const;

        __device__ __host__ inline mat2 inverse() const;

        __device__ __host__ inline mat2 adjoint() const;

        __device__ __host__ inline mat2 cofactor() const;
    };

    __device__ __host__ inline mat2 mat2::identity() {
        return {1, 0,
                0, 1};
    }

    __device__ __host__ inline mat2 mat2::constant(float c) {
        return {c, c,
                c, c};
    }

    __device__ __host__ inline mat2 mat2::reflect_x() {
        return {1, 0,
                0, -1};
    }

    __device__ __host__ inline mat2 mat2::reflect_y() {
        return {-1, 0,
                0, 1};
    }

    __device__ __host__ inline mat2 mat2::rot(float angle) {
        float c = cosf(angle);
        float s = sinf(angle);
        return {c, -s,
                s, c};
    }

    __device__ __host__ inline mat2 mat2::project(float angle) {
        float c = cosf(angle);
        float s = sinf(angle);
        float cs = c * s;
        return {c * c, cs,
                cs, s * s};
    }

    __device__ __host__ inline mat2 mat2::shear_x(float s) {
        return {1, s,
                0, 1};
    }

    __device__ __host__ inline mat2 mat2::shear_y(float s) {
        return {1, 0,
                s, 1};
    }

    __device__ __host__ inline mat2 mat2::operator-() const {
        return {-col0, -col1};
    }

    __device__ __host__ inline vec2 mat2::operator[](int i) const {
        return *(&col0 + i);
    }

    __device__ __host__ inline vec2 &mat2::operator[](int i) {
        return *(&col0 + i);
    }

    __device__ __host__ inline const float &mat2::operator()(int r, int c) const {
        return (*this)[c][r];
    }

    __device__ __host__ inline float &mat2::operator()(int r, int c) {
        return (*this)[c][r];
    }

    __device__ __host__ inline mat2 mat2::operator+(const mat2 &b) const {
        return {col0 + b.col0, col1 + b.col1};
    }

    __device__ __host__ inline mat2 mat2::operator-(const mat2 &b) const {
        return {col0 - b.col0, col1 - b.col1};
    }

    __device__ __host__ inline mat2 mat2::operator*(const mat2 &b) const {
        float m00 = col0.x * b.col0.x + col1.x * b.col0.y;
        float m01 = col0.y * b.col0.x + col1.y * b.col0.y;
        float m10 = col0.x * b.col1.x + col1.x * b.col1.y;
        float m11 = col0.y * b.col1.x + col1.y * b.col1.y;
        return {m00, m01,
                m10, m11};
    }

    __device__ __host__ inline mat2 mat2::operator+(float b) const {
        return {col0 + b, col1 + b};
    }

    __device__ __host__ inline mat2 mat2::operator-(float b) const {
        return {col0 - b, col1 - b};
    }

    __device__ __host__ inline mat2 mat2::operator*(float b) const {
        return {col0 * b, col1 * b};
    }

    __device__ __host__ inline mat2 mat2::operator/(float b) const {
        return {col0 / b, col1 / b};
    }

    __device__ __host__ inline vec2 mat2::operator*(const vec2 &v) const {
        return {col0.x * v.x + col1.x * v.y,
                col0.y * v.x + col1.y * v.y};
    }

    __device__ __host__ inline mat2 mat2::transpose() const {
        return {col0.x, col0.y,
                col1.x, col1.y};
    }

    __device__ __host__ inline double mat2_determinant(double a, double b,
                                                       double c, double d) {
        return a * d - c * b;
    }

    __device__ __host__ inline float mat2::determinant() const {
        return mat2_determinant(col0.x, col1.x,
                                col0.y, col1.y);
    }

    __device__ __host__ inline mat2 mat2::inverse() const {
        return adjoint() / determinant();
    }

    __device__ __host__ inline mat2 mat2::adjoint() const {
        return {col1.y, -col1.x,
                -col0.y, col0.x};
    }

    __device__ __host__ inline mat2 mat2::cofactor() const {
        return {col1.y, -col0.y,
                -col1.x, col0.x};
    }

    __device__ __host__ inline mat2 operator+(float a, const mat2 &b) {
        return {a + b.col0, a + b.col1};
    }

    __device__ __host__ inline mat2 operator-(float a, const mat2 &b) {
        return {a - b.col0, a - b.col1};
    }

    __device__ __host__ inline mat2 operator*(float a, const mat2 &b) {
        return {a * b.col0, a * b.col1};
    }

    __device__ __host__ inline mat2 operator/(float a, const mat2 &b) {
        return {a / b.col0, a / b.col1};
    }
}

#endif //ENGINE24_MAT2_CUH
