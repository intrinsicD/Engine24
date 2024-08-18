//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_MAT3_CUH
#define ENGINE24_MAT3_CUH

#include "vec3.cuh"
#include "mat2.cuh"

namespace Bcg::cuda {
    struct mat2;

    struct mat3 {
        vec3 col0, col1, col2;

        __device__ __host__ inline mat3();

        __device__ __host__ inline mat3(vec3 col0, vec3 col1, vec3 col2);

        __device__ __host__ inline static mat3 identity();

        __device__ __host__ inline static mat3 constant(float c);

        __device__ __host__ inline static mat3 reflect_x();

        __device__ __host__ inline static mat3 reflect_y();

        __device__ __host__ inline static mat3 reflect_z();

        __device__ __host__ inline static mat3 rot(const vec3 axis, float angle);

        __device__ __host__ inline static mat3 scale(const vec3 s);

        __device__ __host__ inline static mat3 project(const vec3 &normal);

        __device__ __host__ inline static mat3 shear_x(float s);

        __device__ __host__ inline static mat3 shear_y(float s);

        __device__ __host__ inline static mat3 shear_z(float s);

        __device__ __host__ inline mat2 left_upper() const;

        __device__ __host__ inline mat2 right_upper() const;

        __device__ __host__ inline mat2 left_lower() const;

        __device__ __host__ inline mat2 right_lower() const;

        __device__ __host__ inline mat3 operator-() const;

        __device__ __host__ inline vec3 operator[](int i) const;

        __device__ __host__ inline vec3 &operator[](int i);

        __device__ __host__ inline const float &operator()(int r, int c) const;

        __device__ __host__ inline float &operator()(int r, int c);

        __device__ __host__ inline mat3 operator+(const mat3 &b) const;

        __device__ __host__ inline mat3 operator-(const mat3 &b) const;

        __device__ __host__ inline mat3 operator*(const mat3 &b) const;

        __device__ __host__ inline mat3 operator+(float b) const;

        __device__ __host__ inline mat3 operator-(float b) const;

        __device__ __host__ inline mat3 operator*(float b) const;

        __device__ __host__ inline mat3 operator/(float b) const;

        __device__ __host__ inline vec3 operator*(const vec3 &v) const;

        __device__ __host__ inline mat3 transpose() const;

        __device__ __host__ inline float determinant() const;

        __device__ __host__ inline mat3 inverse() const;

        __device__ __host__ inline mat3 adjoint() const;

        __device__ __host__ inline mat3 cofactor() const;

        __device__ __host__ inline float norm() const;

        __device__ __host__ inline float trace() const;

        __device__ __host__ inline float geroshin_radius() const;

        //for symmetric positive definite matrices
        __device__ __host__ inline float max_row_sum() const;
    };

    __device__ __host__ inline  mat3::mat3() : col0(), col1(), col2() {

    }

    __device__ __host__ inline  mat3::mat3(vec3 col0, vec3 col1, vec3 col2) : col0(col0), col1(col1), col2(col2) {

    }

    __device__ __host__ inline mat3 mat3::identity() {
        return {{1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}};
    }

    __device__ __host__ inline mat3 mat3::constant(float c) {
        return {{c, c, c},
                {c, c, c},
                {c, c, c}};
    }

    __device__ __host__ inline mat3 mat3::reflect_x() {
        return {{-1, 0, 0},
                {0,  1, 0},
                {0,  0, 1}};
    }

    __device__ __host__ inline mat3 mat3::reflect_y() {
        return {{1, 0,  0},
                {0, -1, 0},
                {0, 0,  1}};
    }

    __device__ __host__ inline mat3 mat3::reflect_z() {
        return {{1, 0, 0},
                {0, 1, 0},
                {0, 0, -1}};
    }

    __device__ __host__ inline mat3 mat3::rot(const vec3 axis, float angle) {
        float c = cosf(angle);
        float s = sinf(angle);
        float t = 1 - c;
        vec3 a = axis.normalized();
        return {{t * a.x * a.x + c,       t * a.x * a.y - s * a.z, t * a.x * a.z + s * a.y},
                {t * a.x * a.y + s * a.z, t * a.y * a.y + c,       t * a.y * a.z - s * a.x},
                {t * a.x * a.z - s * a.y, t * a.y * a.z + s * a.x, t * a.z * a.z + c}};
    }

    __device__ __host__ inline mat3 mat3::scale(const vec3 s) {
        return {{s.x, 0,   0},
                {0,   s.y, 0},
                {0,   0,   s.z}};
    }

    __device__ __host__ inline mat3 mat3::project(const vec3 &normal) {
        vec3 n = normal.normalized();
        return {{1 - n.x * n.x, -n.x * n.y,    -n.x * n.z},
                {-n.y * n.x,    1 - n.y * n.y, -n.y * n.z},
                {-n.z * n.x,    -n.z * n.y,    1 - n.z * n.z}};
    }


    __device__ __host__ inline mat3 mat3::shear_x(float s) {
        return {{1, 0, 0},
                {s, 1, 0},
                {0, 0, 1}};
    }

    __device__ __host__ inline mat3 mat3::shear_y(float s) {
        return {{1, s, 0},
                {0, 1, 0},
                {0, 0, 1}};
    }

    __device__ __host__ inline mat3 mat3::shear_z(float s) {
        return {{1, 0, s},
                {0, 1, 0},
                {0, 0, 1}};
    }

    __device__ __host__ inline mat2 mat3::left_upper() const {
        return {{col0.x, col0.y},
                {col1.x, col1.y}};
    }

    __device__ __host__ inline mat2 mat3::right_upper() const {
        return {{col1.x, col1.y},
                {col2.x, col1.y}};
    }

    __device__ __host__ inline mat2 mat3::left_lower() const {
        return {{col0.y, col0.z},
                {col1.y, col1.z}};
    }

    __device__ __host__ inline mat2 mat3::right_lower() const {
        return {{col1.y, col1.z},
                {col2.y, col1.z}};
    }

    __device__ __host__ inline mat3 mat3::operator-() const {
        return {-col0, -col1, -col2};
    }

    __device__ __host__ inline vec3 mat3::operator[](int i) const {
        //internally vec3 is a vec4 with the last component set to 0
        return (&col0)[i];
    }

    __device__ __host__ inline vec3 &mat3::operator[](int i) {
        return (&col0)[i];
    }

    __device__ __host__ inline const float &mat3::operator()(int r, int c) const {
        return (*this)[c][r];
    }

    __device__ __host__ inline float &mat3::operator()(int r, int c) {
        return (*this)[c][r];
    }

    __device__ __host__ inline mat3 mat3::operator+(const mat3 &b) const {
        return {col0 + b.col0, col1 + b.col1, col2 + b.col2};
    }

    __device__ __host__ inline mat3 mat3::operator-(const mat3 &b) const {
        return {col0 - b.col0, col1 - b.col1, col2 - b.col2};
    }

    __device__ __host__ inline mat3 mat3::operator*(const mat3 &b) const {
        return {col0 * b.col0.x + col1 * b.col0.y + col2 * b.col0.z,
                col0 * b.col1.x + col1 * b.col1.y + col2 * b.col1.z,
                col0 * b.col2.x + col1 * b.col2.y + col2 * b.col2.z};
    }

    __device__ __host__ inline mat3 mat3::operator+(float b) const {
        return {col0 + b, col1 + b, col2 + b};
    }

    __device__ __host__ inline mat3 mat3::operator-(float b) const {
        return {col0 - b, col1 - b, col2 - b};
    }

    __device__ __host__ inline mat3 mat3::operator*(float b) const {
        return {col0 * b, col1 * b, col2 * b};
    }

    __device__ __host__ inline mat3 mat3::operator/(float b) const {
        return {col0 / b, col1 / b, col2 / b};
    }

    __device__ __host__ inline vec3 mat3::operator*(const vec3 &v) const {
        return {col0 * v.x + col1 * v.y + col2 * v.z};
    }

    __device__ __host__ inline mat3 mat3::transpose() const {
        return {{col0.x, col1.x, col2.x},
                {col0.y, col1.y, col2.y},
                {col0.z, col1.z, col2.z}};
    }

    __device__ __host__ inline float mat3_determinant(
            float a00, float a01, float a02,
            float a10, float a11, float a12,
            float a20, float a21, float a22) {
        return a00 * mat2_determinant(a11, a12, a21, a22)
               - a01 * mat2_determinant(a10, a12, a20, a22)
               + a02 * mat2_determinant(a10, a11, a20, a21);
    }

    __device__ __host__ inline float mat3::determinant() const {
        return mat3_determinant(col0.x, col0.y, col0.z, col1.x, col1.y, col1.z, col2.x, col2.y, col2.z);
    }

    __device__ __host__ inline mat3 mat3::inverse() const {
        return transpose() / determinant();
    }

    __device__ __host__ inline mat3 mat3::adjoint() const {
        return mat3{
                // First row of cofactors
                vec3{
                        mat2_determinant(col1.y, col1.z, col2.y, col2.z),
                        -mat2_determinant(col1.x, col1.z, col2.x, col2.z),
                        mat2_determinant(col1.x, col1.y, col2.x, col2.y)
                },
                // Second row of cofactors
                vec3{
                        -mat2_determinant(col0.y, col0.z, col2.y, col2.z),
                        mat2_determinant(col0.x, col0.z, col2.x, col2.z),
                        -mat2_determinant(col0.x, col0.y, col2.x, col2.y)
                },
                // Third row of cofactors
                vec3{
                        mat2_determinant(col0.y, col0.z, col1.y, col1.z),
                        -mat2_determinant(col0.x, col0.z, col1.x, col1.z),
                        mat2_determinant(col0.x, col0.y, col1.x, col1.y)
                }
        }.transpose();
    }

    __device__ __host__ inline mat3 mat3::cofactor() const {
        return adjoint();
    }

    __device__ __host__ inline float mat3::norm() const {
        return sqrtf(col0.length2() + col1.length2() + col2.length2());
    }

    __device__ __host__ inline float mat3::trace() const{
        return col0.x + col1.y + col2.z;
    }

    __device__ __host__ inline float mat3::geroshin_radius() const {
        float radius_col0 = col0.x + fabsf(col0.y) + fabsf(col0.z);
        float radius_col1 = col1.y + fabsf(col1.x) + fabsf(col1.z);
        float radius_col2 = col2.z + fabsf(col2.x) + fabsf(col2.y);
        return fmaxf(radius_col0, fmaxf(radius_col1, radius_col2));
    }

    __device__ __host__ inline float mat3::max_row_sum() const {
        float row0_sum = fabsf(col0.x) + fabsf(col0.y) + fabsf(col0.z);
        float row1_sum = fabsf(col1.x) + fabsf(col1.y) + fabsf(col1.z);
        float row2_sum = fabsf(col2.x) + fabsf(col2.y) + fabsf(col2.z);
        return fmaxf(row0_sum, fmaxf(row1_sum, row2_sum));
    }

    __device__ __host__ inline mat3 operator-(float a, const mat3 &b) {
        return {a - b.col0, a - b.col1, a - b.col2};
    }

    __device__ __host__ inline mat3 operator*(float a, const mat3 &b) {
        return {a * b.col0, a * b.col1, a * b.col2};
    }

    __device__ __host__ inline mat3 operator/(float a, const mat3 &b) {
        return {a / b.col0, a / b.col1, a / b.col2};
    }
}

#endif //ENGINE24_MAT3_CUH
