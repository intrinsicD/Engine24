//
// Created by alex on 29.05.25.
//

#ifndef ENGINE24_MAT_VEC_CUH
#define ENGINE24_MAT_VEC_CUH

#include <cassert>              // for assert()
#include <cmath>                // for sqrtf
#include <type_traits>          // for std::is_trivially_copyable
#include <vector_types.h>       // for float2, float3, float4
#include <vector_functions.h>   // for make_float2, make_float3, make_float4
#include <cuda_runtime.h>       // for __host__, __device__

namespace Bcg {
    struct vec2 {
        float data[2];

        __host__ __device__
        constexpr vec2() noexcept: data{0.0f, 0.0f} {}

        __host__ __device__
        explicit vec2(float s) : data{s, s} {}

        __host__ __device__
        vec2(float x, float y) : data{x, y} {}

        __host__ __device__
        explicit vec2(const float2 &v) : data{v.x, v.y} {}

        __host__ __device__
        float &operator[](int index) {
            return data[index];
        }

        __host__ __device__
        const float &operator[](int index) const {
            return data[index];
        }

        __host__ __device__
        operator float2() const {
            return make_float2(data[0], data[1]);
        }

        __host__ __device__ __forceinline__
        vec2 operator-() const {
            return vec2(-data[0], -data[1]);
        }

        __host__ __device__ __forceinline__
        vec2 operator+(const vec2 &other) const {
            return vec2(data[0] + other.data[0], data[1] + other.data[1]);
        }

        __host__ __device__ __forceinline__
        vec2 operator-(const vec2 &other) const {
            return vec2(data[0] - other.data[0], data[1] - other.data[1]);
        }

        __host__ __device__ __forceinline__
        vec2 operator*(float scalar) const {
            return vec2(data[0] * scalar, data[1] * scalar);
        }

        __host__ __device__ __forceinline__
        vec2 operator/(float scalar) const {
            return vec2(data[0] / scalar, data[1] / scalar);
        }

        __host__ __device__ __forceinline__
        vec2 operator*(const vec2 &other) const {
            return vec2(data[0] * other.data[0], data[1] * other.data[1]);
        }

        __host__ __device__ __forceinline__
        vec2 operator/(const vec2 &other) const {
            return vec2(data[0] / other.data[0], data[1] / other.data[1]);
        }
    };

    struct vec3 {
        float data[3];

        __host__ __device__
        constexpr vec3() noexcept: data{0.0f, 0.0f, 0.0f} {}

        __host__ __device__
        explicit vec3(float s) : data{s, s, s} {}

        __host__ __device__
        vec3(float x, float y, float z) : data{x, y, z} {}

        __host__ __device__
        explicit vec3(const float3 &v) : data{v.x, v.y, v.z} {}

        __host__ __device__
        float &operator[](int index) {
            return data[index];
        }

        __host__ __device__
        const float &operator[](int index) const {
            return data[index];
        }

        __host__ __device__
        operator float3() const {
            return make_float3(data[0], data[1], data[2]);
        }

        __host__ __device__ __forceinline__
        vec3 operator-() const {
            return vec3(-data[0], -data[1], -data[2]);
        }

        __host__ __device__ __forceinline__
        vec3 operator+(const vec3 &other) const {
            return vec3(data[0] + other.data[0], data[1] + other.data[1], data[2] + other.data[2]);
        }

        __host__ __device__ __forceinline__
        vec3 operator-(const vec3 &other) const {
            return vec3(data[0] - other.data[0], data[1] - other.data[1], data[2] - other.data[2]);
        }

        __host__ __device__ __forceinline__
        vec3 operator*(float scalar) const {
            return vec3(data[0] * scalar, data[1] * scalar, data[2] * scalar);
        }

        __host__ __device__ __forceinline__
        vec3 operator/(float scalar) const {
            return vec3(data[0] / scalar, data[1] / scalar, data[2] / scalar);
        }

        __host__ __device__ __forceinline__
        vec3 operator*(const vec3 &other) const {
            return vec3(data[0] * other.data[0], data[1] * other.data[1], data[2] * other.data[2]);
        }

        __host__ __device__ __forceinline__
        vec3 operator/(const vec3 &other) const {
            return vec3(data[0] / other.data[0], data[1] / other.data[1], data[2] / other.data[2]);
        }
    };

    struct alignas(16) vec4 {
        float data[4];

        __host__ __device__
        constexpr vec4() noexcept: data{0.0f, 0.0f, 0.0f, 0.0f} {}

        __host__ __device__
        explicit vec4(float s) : data{s, s, s, s} {}

        __host__ __device__
        vec4(float x, float y, float z, float w) : data{x, y, z, w} {}

        __host__ __device__
        explicit vec4(const float4 &v) : data{v.x, v.y, v.z, v.w} {}

        __host__ __device__
        float &operator[](int index) {
            return data[index];
        }

        __host__ __device__
        const float &operator[](int index) const {
            return data[index];
        }

        __host__ __device__
        operator float4() const {
            return make_float4(data[0], data[1], data[2], data[3]);
        }

        __host__ __device__ __forceinline__
        vec4 operator-() const {
            return vec4(-data[0], -data[1], -data[2], -data[3]);
        }

        __host__ __device__ __forceinline__
        vec4 operator+(const vec4 &other) const {
            return vec4(data[0] + other.data[0], data[1] + other.data[1],
                        data[2] + other.data[2], data[3] + other.data[3]);
        }

        __host__ __device__ __forceinline__
        vec4 operator-(const vec4 &other) const {
            return vec4(data[0] - other.data[0], data[1] - other.data[1],
                        data[2] - other.data[2], data[3] - other.data[3]);
        }

        __host__ __device__ __forceinline__
        vec4 operator*(float scalar) const {
            return vec4(data[0] * scalar, data[1] * scalar, data[2] * scalar, data[3] * scalar);
        }

        __host__ __device__ __forceinline__
        vec4 operator/(float scalar) const {
            return vec4(data[0] / scalar, data[1] / scalar, data[2] / scalar, data[3] / scalar);
        }

        __host__ __device__ __forceinline__
        vec4 operator*(const vec4 &other) const {
            return vec4(data[0] * other.data[0], data[1] * other.data[1],
                        data[2] * other.data[2], data[3] * other.data[3]);
        }

        __host__ __device__ __forceinline__
        vec4 operator/(const vec4 &other) const {
            return vec4(data[0] / other.data[0], data[1] / other.data[1],
                        data[2] / other.data[2], data[3] / other.data[3]);
        }
    };

    struct mat2 {
        vec2 cols[2];

        __host__ __device__
        constexpr mat2() : cols{{},
                                {}} {};

        __host__ __device__
        explicit mat2(float s) : cols{vec2(s, 0), vec2(0, s)} {}

        __host__ __device__
        mat2(const vec2 &x, const vec2 &y) : cols{x, y} {}

        __host__ __device__
        mat2(float m00, float m01,
             float m10, float m11)
                : cols{vec2(m00, m10), vec2(m01, m11)} {}

        __host__ __device__
        const vec2 &operator[](int index) const {
            return cols[index];
        }

        __host__ __device__
        vec2 &operator[](int index) {
            return cols[index];
        }

        __host__ __device__
        float &operator()(int row, int col) {
            return cols[col][row];
        }

        __host__ __device__
        const float &operator()(int row, int col) const {
            return cols[col][row];
        }

        __host__ __device__ __forceinline__
        mat2 operator*(float scalar) const {
            mat2 result;
            result.cols[0] = this->cols[0] * scalar; // Uses vec2::operator*(float)
            result.cols[1] = this->cols[1] * scalar;
            return result;
        }

        __host__ __device__ __forceinline__
        mat2 operator/(float scalar) const {
            mat2 result;
            result.cols[0] = this->cols[0] / scalar; // Uses vec2::operator*(float)
            result.cols[1] = this->cols[1] / scalar;
            return result;
        }

        __host__ __device__ __forceinline__
        vec2 operator*(const vec2 &v) const {
            vec2 result{};
            int N = 2;
            for (int row = 0; row < N; ++row) {
                result[row] = 0;
                for (int k = 0; k < N; ++k) {
                    result[row] += (*this)(row, k) * v[k];
                }
            }
            return result;
        }

        __host__ __device__ __forceinline__
        mat2 operator*(const mat2 &o) const noexcept {
            mat2 r{};
            for (int row = 0; row < 2; ++row) {
                for (int col = 0; col < 2; ++col) {
                    float sum = 0.0f;
                    for (int k = 0; k < 2; ++k) {
                        sum += (*this)(row, k) * o(k, col);
                    }
                    r(row, col) = sum;
                }
            }
            return r;
        }

        __host__ __device__ __forceinline__
        mat2 operator+(const mat2 &other) const {
            mat2 result{};
            result.cols[0] = this->cols[0] + other.cols[0]; // Uses vec2::operator+
            result.cols[1] = this->cols[1] + other.cols[1];
            return result;
        }

        __host__ __device__ __forceinline__
        mat2 operator-(const mat2 &other) const {
            mat2 result{};
            result.cols[0] = this->cols[0] - other.cols[0]; // Uses vec2::operator+
            result.cols[1] = this->cols[1] - other.cols[1];
            return result;
        }
    };

    struct mat3 {
        vec3 cols[3];

        __host__ __device__
        constexpr mat3() : cols{{},
                                {},
                                {}} {};

        __host__ __device__
        explicit mat3(float s) : cols{vec3(s, 0, 0), vec3(0, s, 0), vec3(0, 0, s)} {}

        __host__ __device__
        mat3(const vec3 &x, const vec3 &y, const vec3 &z) : cols{x, y, z} {}

        __host__ __device__
        mat3(float m00, float m01, float m02,
             float m10, float m11, float m12,
             float m20, float m21, float m22)
                : cols{vec3(m00, m10, m20), vec3(m01, m11, m21), vec3(m02, m12, m22)} {}

        __host__ __device__
        const vec3 &operator[](int index) const {
            return cols[index];
        }

        __host__ __device__
        vec3 &operator[](int index) {
            return cols[index];
        }

        __host__ __device__
        float &operator()(int row, int col) {
            return cols[col][row];
        }

        __host__ __device__
        const float &operator()(int row, int col) const {
            return cols[col][row];
        }

        __host__ __device__ __forceinline__
        mat3 operator*(float scalar) const {
            mat3 result{};
            for (int col = 0; col < 3; ++col) {
                for (int row = 0; row < 3; ++row) {
                    result(col, row) = cols[col][row] * scalar;
                }
            }
            return result;
        }

        __host__ __device__ __forceinline__
        mat3 operator/(float scalar) const {
            assert(scalar != 0.0f);
            mat3 result{};
            for (int col = 0; col < 3; ++col) {
                for (int row = 0; row < 3; ++row) {
                    result(col, row) = cols[col][row] / scalar;
                }
            }
            return result;
        }

        __host__ __device__ __forceinline__
        vec3 operator*(const vec3 &v) const {
            vec3 result{};
            int N = 3;
            for (int row = 0; row < N; ++row) {
                result[row] = 0;
                for (int k = 0; k < N; ++k) {
                    result[row] += (*this)(row, k) * v[k];
                }
            }
            return result;
        }

        __host__ __device__ __forceinline__
        mat3 operator*(const mat3 &o) const noexcept {
            mat3 r{};
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += (*this)(row, k) * o(k, col);
                    }
                    r(row, col) = sum;
                }
            }
            return r;
        }

        __host__ __device__ __forceinline__
        mat3 operator+(const mat3 &other) const {
            mat3 result{};
            result.cols[0] = this->cols[0] + other.cols[0]; // Uses vec2::operator+
            result.cols[1] = this->cols[1] + other.cols[1];
            result.cols[2] = this->cols[2] + other.cols[2];
            return result;
        }

        __host__ __device__ __forceinline__
        mat3 operator-(const mat3 &other) const {
            mat3 result{};
            result.cols[0] = this->cols[0] - other.cols[0]; // Uses vec2::operator+
            result.cols[1] = this->cols[1] - other.cols[1];
            result.cols[2] = this->cols[2] - other.cols[2];
            return result;
        }
    };

    struct mat4 {
        vec4 cols[4];

        __host__ __device__
        constexpr mat4() : cols{{},
                                {},
                                {},
                                {}} {};

        __host__ __device__
        explicit mat4(float s) : cols{vec4(s, 0, 0, 0), vec4(0, s, 0, 0), vec4(0, 0, s, 0),
                                      vec4(0, 0, 0, s)} {}

        __host__ __device__
        mat4(const vec4 &x, const vec4 &y, const vec4 &z, const vec4 &w) : cols{x, y, z, w} {}

        __host__ __device__
        mat4(float m00, float m01, float m02, float m03,
             float m10, float m11, float m12, float m13,
             float m20, float m21, float m22, float m23,
             float m30, float m31, float m32, float m33)
                : cols{vec4(m00, m10, m20, m30), vec4(m01, m11, m21, m31),
                       vec4(m02, m12, m22, m32), vec4(m03, m13, m23, m33)} {}

        __host__ __device__
        const vec4 &operator[](int index) const {
            return cols[index];
        }

        __host__ __device__
        vec4 &operator[](int index) {
            return cols[index];
        }

        __host__ __device__
        float &operator()(int row, int col) {
            return cols[col][row];
        }

        __host__ __device__
        const float &operator()(int row, int col) const {
            return cols[col][row];
        }

        __host__ __device__ __forceinline__
        mat4 operator*(float scalar) const {
            mat4 result{};
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 4; ++row) {
                    result(col, row) = cols[col][row] * scalar;
                }
            }
            return result;
        }

        __host__ __device__ __forceinline__
        vec4 operator*(const vec4 &v) const {
            vec4 result{};
            int N = 4;
            for (int row = 0; row < N; ++row) {
                result[row] = 0;
                for (int k = 0; k < N; ++k) {
                    result[row] += (*this)(row, k) * v[k];
                }
            }
            return result;
        }

        __host__ __device__ __forceinline__
        mat4 operator/(float scalar) const {
            assert(scalar != 0.0f);
            mat4 result{};
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 4; ++row) {
                    result(col, row) = cols[col][row] / scalar;
                }
            }
            return result;
        }

        __host__ __device__ __forceinline__
        mat4 operator*(const mat4 &o) const noexcept {
            mat4 r{};
            // for each output row and column…
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    float sum = 0.0f;
                    // dot the row of *this* with the column of o
                    for (int k = 0; k < 4; ++k) {
                        sum += (*this)(row, k) * o(k, col);
                    }
                    r(row, col) = sum;   // <<-- correct indexing
                }
            }
            return r;
        }

        __host__ __device__ __forceinline__
        mat4 operator+(const mat4 &other) const {
            mat4 result{};
            result.cols[0] = this->cols[0] + other.cols[0]; // Uses vec2::operator+
            result.cols[1] = this->cols[1] + other.cols[1];
            result.cols[2] = this->cols[2] + other.cols[2];
            result.cols[3] = this->cols[3] + other.cols[3];
            return result;
        }

        __host__ __device__ __forceinline__
        mat4 operator-(const mat4 &other) const {
            mat4 result{};
            result.cols[0] = this->cols[0] - other.cols[0]; // Uses vec2::operator+
            result.cols[1] = this->cols[1] - other.cols[1];
            result.cols[2] = this->cols[2] - other.cols[2];
            result.cols[3] = this->cols[3] - other.cols[3];
            return result;
        }
    };

    __host__ __device__ __forceinline__
    vec2 operator*(float scalar, const vec2 &v) noexcept {
        return vec2(scalar * v[0], scalar * v[1]);
    }

    __host__ __device__ __forceinline__
    vec3 operator*(float scalar, const vec3 &v) noexcept {
        return vec3(scalar * v[0], scalar * v[1], scalar * v[2]);
    }

    __host__ __device__ __forceinline__
    vec4 operator*(float scalar, const vec4 &v) noexcept {
        return vec4(scalar * v[0], scalar * v[1], scalar * v[2], scalar * v[3]);
    }

    __host__ __device__ __forceinline__
    mat2 operator*(float scalar, const mat2 &m) noexcept {
        return mat2(scalar * m.cols[0][0], scalar * m.cols[0][1],
                    scalar * m.cols[1][0], scalar * m.cols[1][1]);
    }

    __host__ __device__ __forceinline__
    mat3 operator*(float scalar, const mat3 &m) noexcept {
        return mat3(scalar * m.cols[0][0], scalar * m.cols[0][1], scalar * m.cols[0][2],
                    scalar * m.cols[1][0], scalar * m.cols[1][1], scalar * m.cols[1][2],
                    scalar * m.cols[2][0], scalar * m.cols[2][1], scalar * m.cols[2][2]);
    }

    __host__ __device__ __forceinline__
    mat4 operator*(float scalar, const mat4 &m) noexcept {
        return mat4(scalar * m.cols[0][0], scalar * m.cols[0][1], scalar * m.cols[0][2], scalar * m.cols[0][3],
                    scalar * m.cols[1][0], scalar * m.cols[1][1], scalar * m.cols[1][2], scalar * m.cols[1][3],
                    scalar * m.cols[2][0], scalar * m.cols[2][1], scalar * m.cols[2][2], scalar * m.cols[2][3],
                    scalar * m.cols[3][0], scalar * m.cols[3][1], scalar * m.cols[3][2], scalar * m.cols[3][3]);
    }


    __host__ __device__ __forceinline__
    float dot(const vec2 &a, const vec2 &b) noexcept {
        return a[0] * b[0] + a[1] * b[1];
    }

    __host__ __device__ __forceinline__
    float dot(const vec3 &a, const vec3 &b) noexcept {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    __host__ __device__ __forceinline__
    float dot(const vec4 &a, const vec4 &b) noexcept {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    }

    __host__ __device__ __forceinline__
    vec3 cross(const vec3 &a, const vec3 &b) noexcept {
        return vec3(
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
        );
    }

    __host__ __device__ __forceinline__
    float length(const vec2 &v) noexcept {
        return sqrtf(dot(v, v));
    }

    __host__ __device__ __forceinline__
    float length(const vec3 &v) noexcept {
        return sqrtf(dot(v, v));
    }

    __host__ __device__ __forceinline__
    float length(const vec4 &v) noexcept {
        return sqrtf(dot(v, v));
    }

    __host__ __device__ __forceinline__
    vec2 normalize(const vec2 &v) noexcept {
        float len = length(v);
        return len > 0 ? v / len : vec2(0.0f);
    }

    __host__ __device__ __forceinline__
    vec3 normalize(const vec3 &v) noexcept {
        float len = length(v);
        return len > 0 ? v / len : vec3(0.0f);
    }

    __host__ __device__ __forceinline__
    vec4 normalize(const vec4 &v) noexcept {
        float len = length(v);
        return len > 0 ? v / len : vec4(0.0f);
    }

    __host__ __device__ __forceinline__
    mat2 transpose(const mat2 &m) noexcept {
        return mat2(
                m.cols[0][0], m.cols[1][0],
                m.cols[0][1], m.cols[1][1]
        );
    }

    __host__ __device__ __forceinline__
    mat3 transpose(const mat3 &m) noexcept {
        return mat3(
                m.cols[0][0], m.cols[1][0], m.cols[2][0],
                m.cols[0][1], m.cols[1][1], m.cols[2][1],
                m.cols[0][2], m.cols[1][2], m.cols[2][2]
        );
    }

    __host__ __device__ __forceinline__
    mat4 transpose(const mat4 &m) noexcept {
        return mat4(
                m.cols[0][0], m.cols[1][0], m.cols[2][0], m.cols[3][0],
                m.cols[0][1], m.cols[1][1], m.cols[2][1], m.cols[3][1],
                m.cols[0][2], m.cols[1][2], m.cols[2][2], m.cols[3][2],
                m.cols[0][3], m.cols[1][3], m.cols[2][3], m.cols[3][3]
        );
    }

    __host__ __device__ __forceinline__
    mat3 outer(const vec3 &a, const vec3 &b) noexcept {
        return mat3(
                a[0] * b[0], a[0] * b[1], a[0] * b[2],
                a[1] * b[0], a[1] * b[1], a[1] * b[2],
                a[2] * b[0], a[2] * b[1], a[2] * b[2]
        );
    }

    __host__ __device__ __forceinline__
    mat4 outer(const vec4 &a, const vec4 &b) noexcept {
        return mat4(
                a[0] * b[0], a[0] * b[1], a[0] * b[2], a[0] * b[3],
                a[1] * b[0], a[1] * b[1], a[1] * b[2], a[1] * b[3],
                a[2] * b[0], a[2] * b[1], a[2] * b[2], a[2] * b[3],
                a[3] * b[0], a[3] * b[1], a[3] * b[2], a[3] * b[3]
        );
    }

    __host__ __device__ __forceinline__
    float trace(const mat2 &m) noexcept {
        return m.cols[0][0] + m.cols[1][1];
    }

    __host__ __device__ __forceinline__
    float trace(const mat3 &m) noexcept {
        return m.cols[0][0] + m.cols[1][1] + m.cols[2][2];
    }

    __host__ __device__ __forceinline__
    float trace(const mat4 &m) noexcept {
        return m.cols[0][0] + m.cols[1][1] + m.cols[2][2] + m.cols[3][3];
    }

    inline namespace helper {
        __host__ __device__ __forceinline__
        float minor(const mat3 &m, int r0, int r1, int r2, int c0, int c1, int c2) noexcept {
            return m(r0, c0) * (m(r1, c1) * m(r2, c2) - m(r1, c2) * m(r2, c1)) -
                   m(r0, c1) * (m(r1, c0) * m(r2, c2) - m(r1, c2) * m(r2, c0)) +
                   m(r0, c2) * (m(r1, c0) * m(r2, c1) - m(r1, c1) * m(r2, c0));
        }

        __host__ __device__ __forceinline__
        float minor(const mat4 &m, int r0, int r1, int r2, int c0, int c1, int c2) noexcept {
            return m(r0, c0) * (m(r1, c1) * m(r2, c2) - m(r1, c2) * m(r2, c1)) -
                   m(r0, c1) * (m(r1, c0) * m(r2, c2) - m(r1, c2) * m(r2, c0)) +
                   m(r0, c2) * (m(r1, c0) * m(r2, c1) - m(r1, c1) * m(r2, c0));
        }
    }

    __host__ __device__ __forceinline__
    float determinant(const mat2 &m) noexcept {
        return m.cols[0][0] * m.cols[1][1] - m.cols[0][1] * m.cols[1][0];
    }

    __host__ __device__ __forceinline__
    float determinant(const mat3 &m) noexcept {
        return m.cols[0][0] * (m.cols[1][1] * m.cols[2][2] - m.cols[1][2] * m.cols[2][1]) -
               m.cols[0][1] * (m.cols[1][0] * m.cols[2][2] - m.cols[1][2] * m.cols[2][0]) +
               m.cols[0][2] * (m.cols[1][0] * m.cols[2][1] - m.cols[1][1] * m.cols[2][0]);
    }

    __host__ __device__ __forceinline__
    float determinant(const mat4 &m) noexcept {
        return m(0, 0) * helper::minor(m, 1, 2, 3, 1, 2, 3) -
               m(0, 1) * helper::minor(m, 1, 2, 3, 0, 2, 3) +
               m(0, 2) * helper::minor(m, 1, 2, 3, 0, 1, 3) -
               m(0, 3) * helper::minor(m, 1, 2, 3, 0, 1, 2);
    }

    // -------------------- 2×2 --------------------
    __host__ __device__ __forceinline__
    mat2 inverse(const mat2 &m) noexcept {
        float det = determinant(m);
        assert(det != 0.0f);
        // adjugate of [ a b ; c d ] is [ d -b ; -c a ]
        return mat2(
                m(1,1) / det, -m(0,1) / det,
                -m(1,0) / det,  m(0,0) / det
        );
    }

// -------------------- 3×3 --------------------
// cofactor(i,j) =  (−1)^(i+j) * minor of element at row i, col j
// adjugate = transpose of cofactor‐matrix
    __host__ __device__ __forceinline__
    mat3 inverse(const mat3 &m) noexcept {
        float det = determinant(m); // Assumes determinant(mat3) is correct (it appears to be)
        assert(det != 0.0f);
        float invDet = 1.0f / det;

        mat3 adj;
        // Cofactors (elements of the adjugate matrix are transposed cofactors)
        // C_ij = (-1)^(i+j) * M_ij
        // Adj_ij = C_ji

        adj(0,0) =  (m(1,1) * m(2,2) - m(1,2) * m(2,1));
        adj(0,1) = -(m(0,1) * m(2,2) - m(0,2) * m(2,1));
        adj(0,2) =  (m(0,1) * m(1,2) - m(0,2) * m(1,1));

        adj(1,0) = -(m(1,0) * m(2,2) - m(1,2) * m(2,0));
        adj(1,1) =  (m(0,0) * m(2,2) - m(0,2) * m(2,0));
        adj(1,2) = -(m(0,0) * m(1,2) - m(0,2) * m(1,0));

        adj(2,0) =  (m(1,0) * m(2,1) - m(1,1) * m(2,0));
        adj(2,1) = -(m(0,0) * m(2,1) - m(0,1) * m(2,0));
        adj(2,2) =  (m(0,0) * m(1,1) - m(0,1) * m(1,0));

        // Multiply adjugate by 1/det
        // This can reuse the mat3 operator*(float scalar) if you define it,
        // or implement it directly:
        mat3 result;
        result.cols[0] = adj.cols[0] * invDet;
        result.cols[1] = adj.cols[1] * invDet;
        result.cols[2] = adj.cols[2] * invDet;
        return result;
    }

// -------------------- 4×4 --------------------
// we already have helper::minor(m, r0,r1,r2, c0,c1,c2)
// to generate every cofactor.
// adjugate is transpose of cofactor matrix.
    __host__ __device__ __forceinline__
    mat4 inverse(const mat4 &m) noexcept {
        float det = determinant(m);
        assert(det != 0.0f);

        // row 0
        float c00 =  helper::minor(m,1,2,3, 1,2,3);
        float c01 = -helper::minor(m,1,2,3, 0,2,3);
        float c02 =  helper::minor(m,1,2,3, 0,1,3);
        float c03 = -helper::minor(m,1,2,3, 0,1,2);

        // row 1
        float c10 = -helper::minor(m,0,2,3, 1,2,3);
        float c11 =  helper::minor(m,0,2,3, 0,2,3);
        float c12 = -helper::minor(m,0,2,3, 0,1,3);
        float c13 =  helper::minor(m,0,2,3, 0,1,2);

        // row 2
        float c20 =  helper::minor(m,0,1,3, 1,2,3);
        float c21 = -helper::minor(m,0,1,3, 0,2,3);
        float c22 =  helper::minor(m,0,1,3, 0,1,3);
        float c23 = -helper::minor(m,0,1,3, 0,1,2);

        // row 3
        float c30 = -helper::minor(m,0,1,2, 1,2,3);
        float c31 =  helper::minor(m,0,1,2, 0,2,3);
        float c32 = -helper::minor(m,0,1,2, 0,1,3);
        float c33 =  helper::minor(m,0,1,2, 0,1,2);

        // adjugate = transpose of cofactor matrix:
        return mat4(
                c00/det, c10/det, c20/det, c30/det,
                c01/det, c11/det, c21/det, c31/det,
                c02/det, c12/det, c22/det, c32/det,
                c03/det, c13/det, c23/det, c33/det
        );
    }

    static_assert(std::is_trivially_copyable<vec2>::value, "vec2 must be POD");
    static_assert(std::is_trivially_copyable<vec3>::value, "vec3 must be POD");
    static_assert(std::is_trivially_copyable<vec4>::value, "vec4 must be POD");
    static_assert(std::is_trivially_copyable<mat2>::value, "mat2 must be POD");
    static_assert(std::is_trivially_copyable<mat3>::value, "mat3 must be POD");
    static_assert(std::is_trivially_copyable<mat4>::value, "mat4 must be POD");
}

#endif //ENGINE24_MAT_VEC_CUH
