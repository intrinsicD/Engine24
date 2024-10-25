//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_VEC4_CUH
#define ENGINE24_VEC4_CUH

#include "vec3.cuh"

namespace Bcg::cuda {
    struct vec4 {
        float x, y, z, w;

        __device__ __host__ inline vec4() : x(0), y(0), z(0), w(0) {

        }

        __device__ __host__ inline vec4(const vec2 &v) : x(v.x), y(v.y), z(0), w(0) {

        }

        __device__ __host__ inline vec4(const vec3 &v) : x(v.x), y(v.y), z(v.z), w(0) {

        }

        __device__ __host__ inline vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {

        }

        __device__ __host__ inline static vec4 constant(float c);

        __device__ __host__ inline static vec4 unit(int i);

        __device__ __host__ inline vec4 operator-() const;

        __device__ __host__ inline float operator[](int i) const;

        __device__ __host__ inline float &operator[](int i);

        __device__ __host__ inline float dot(const vec4 &b) const;

        __device__ __host__ inline float length2() const;

        __device__ __host__ inline float length() const;

        __device__ __host__ inline vec4 normalized() const;

        __device__ __host__ inline vec4 operator+(const vec4 &b) const;

        __device__ __host__ inline vec4 operator-(const vec4 &b) const;

        __device__ __host__ inline vec4 operator*(const vec4 &b) const;

        __device__ __host__ inline vec4 operator/(const vec4 &b) const;

        __device__ __host__ inline vec4 operator+(float b) const;

        __device__ __host__ inline vec4 operator-(float b) const;

        __device__ __host__ inline vec4 operator*(float b) const;

        __device__ __host__ inline vec4 operator/(float b) const;
    };

    __device__ __host__ inline vec4 vec4::constant(float c) {
        return {c, c, c, c};
    }

    __device__ __host__ inline vec4 vec4::unit(int i) {
        return {float(i == 0), float(i == 1), float(i == 2), float(i == 3)};
    }

    __device__ __host__ inline vec4 vec4::operator-() const {
        return {-x, -y, -z, -w};
    }

    __device__ __host__ inline float vec4::operator[](int i) const {
        return (&x)[i];
    }

    __device__ __host__ inline float &vec4::operator[](int i) {
        return (&x)[i];
    }

    __device__ __host__ inline float vec4::dot(const vec4 &b) const {
        return x * b.x + y * b.y + z * b.z + w * b.w;
    }

    __device__ __host__ inline float vec4::length2() const {
        return dot(*this);
    }

    __device__ __host__ inline float vec4::length() const {
        return sqrtf(length2());
    }

    __device__ __host__ inline vec4 vec4::normalized() const {
        return {*this / length()};
    }

    __device__ __host__ inline vec4 vec4::operator+(const vec4 &b) const {
        return {x + b.x, y + b.y, z + b.z, w + b.w};
    }

    __device__ __host__ inline vec4 vec4::operator-(const vec4 &b) const {
        return {x - b.x, y - b.y, z - b.z, w - b.w};
    }

    __device__ __host__ inline vec4 vec4::operator*(const vec4 &b) const {
        return {x * b.x, y * b.y, z * b.z, w * b.w};
    }

    __device__ __host__ inline vec4 vec4::operator/(const vec4 &b) const {
        return {x / b.x, y / b.y, z / b.z, w / b.w};
    }

    __device__ __host__ inline vec4 vec4::operator+(float b) const {
        return {x + b, y + b, z + b, w + b};
    }

    __device__ __host__ inline vec4 vec4::operator-(float b) const {
        return {x - b, y - b, z - b, w - b};
    }

    __device__ __host__ inline vec4 vec4::operator*(float b) const {
        return {x * b, y * b, z * b, w * b};
    }

    __device__ __host__ inline vec4 vec4::operator/(float b) const {
        return {x / b, y / b, z / b, w / b};
    }

    __device__ __host__ inline vec4 operator+(float a, const vec4 &b) {
        return {a + b.x, a + b.y, a + b.z, a + b.w};
    }

    __device__ __host__ inline vec4 operator-(float a, const vec4 &b) {
        return {a - b.x, a - b.y, a - b.z, a - b.w};
    }

    __device__ __host__ inline vec4 operator*(float a, const vec4 &b) {
        return {a * b.x, a * b.y, a * b.z, a * b.w};
    }

    __device__ __host__ inline vec4 operator/(float a, const vec4 &b) {
        return {a / b.x, a / b.y, a / b.z, a / b.w};
    }
}

#endif //ENGINE24_VEC4_CUH
