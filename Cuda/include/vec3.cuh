//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_VEC3_CUH
#define ENGINE24_VEC3_CUH

#include "vec2.cuh"

namespace Bcg::cuda {
    struct vec3 {
        float x, y, z;

        __device__ __host__ inline vec3();

        __device__ __host__ inline vec3(const vec2 &v);

        __device__ __host__ inline vec3(float x, float y, float z);

        __device__ __host__ inline static vec3 constant(float c);

        __device__ __host__ inline static vec3 unit(int i);

        __device__ __host__ inline vec3 operator-() const;

        __device__ __host__ inline float operator[](int i) const;

        __device__ __host__ inline float &operator[](int i);

        __device__ __host__ inline float dot(const vec3 &b) const;

        __device__ __host__ inline float length2() const;

        __device__ __host__ inline float length() const;

        __device__ __host__ inline vec3 normalized() const;

        __device__ __host__ inline vec3 cross(const vec3 &b) const;

        __device__ __host__ inline vec3 operator+(const vec3 &b) const;

        __device__ __host__ inline vec3 operator-(const vec3 &b) const;

        __device__ __host__ inline vec3 operator*(const vec3 &b) const;

        __device__ __host__ inline vec3 operator/(const vec3 &b) const;

        __device__ __host__ inline vec3 operator+(float b) const;

        __device__ __host__ inline vec3 operator-(float b) const;

        __device__ __host__ inline vec3 operator*(float b) const;

        __device__ __host__ inline vec3 operator/(float b) const;
    };

    __device__ __host__ inline vec3::vec3() : x(0), y(0), z(0) {

    }

    __device__ __host__ inline vec3::vec3(const vec2 &v) : x(v.x), y(v.y), z(0) {

    }

    __device__ __host__ inline vec3::vec3(float x, float y, float z) : x(x), y(y), z(z) {

    }

    __device__ __host__ inline vec3 vec3::constant(float c) {
        return {c, c, c};
    }

    __device__ __host__ inline vec3 vec3::unit(int i) {
        return {float(i == 0), float(i == 1), float(i == 2)};
    }

    __device__ __host__ inline vec3 vec3::operator-() const {
        return {-x, -y, -z};
    }

    __device__ __host__ inline float vec3::operator[](int i) const {
        return (&x)[i];
    }

    __device__ __host__ inline float &vec3::operator[](int i) {
        return (&x)[i];
    }

    __device__ __host__ inline float vec3::dot(const vec3 &b) const {
        return x * b.x + y * b.y + z * b.z;
    }

    __device__ __host__ inline float vec3::length2() const {
        return dot(*this);
    }

    __device__ __host__ inline float vec3::length() const {
        return sqrtf(length2());
    }

    __device__ __host__ inline vec3 vec3::normalized() const {
        return {*this / length()};
    }

    __device__ __host__ inline vec3 vec3::cross(const vec3 &b) const {
        return {y * b.z - z * b.y,
                z * b.x - x * b.z,
                x * b.y - y * b.x};
    }

    __device__ __host__ inline vec3 vec3::operator+(const vec3 &b) const {
        return {x + b.x, y + b.y, z + b.z};
    }

    __device__ __host__ inline vec3 vec3::operator-(const vec3 &b) const {
        return {x - b.x, y - b.y, z - b.z};
    }

    __device__ __host__ inline vec3 vec3::operator*(const vec3 &b) const {
        return {x * b.x, y * b.y, z * b.z};
    }

    __device__ __host__ inline vec3 vec3::operator/(const vec3 &b) const {
        return {x / b.x, y / b.y, z / b.z};
    }

    __device__ __host__ inline vec3 vec3::operator+(float b) const {
        return {x + b, y + b, z + b};
    }

    __device__ __host__ inline vec3 vec3::operator-(float b) const {
        return {x - b, y - b, z - b};
    }

    __device__ __host__ inline vec3 vec3::operator*(float b) const {
        return {x * b, y * b, z * b};
    }

    __device__ __host__ inline vec3 vec3::operator/(float b) const {
        return {x / b, y / b, z / b};
    }

    __device__ __host__ inline vec3 operator+(float a, const vec3 &b) {
        return {a + b.x, a + b.y, a + b.z};
    }

    __device__ __host__ inline vec3 operator-(float a, const vec3 &b) {
        return {a - b.x, a - b.y, a - b.z};
    }

    __device__ __host__ inline vec3 operator*(float a, const vec3 &b) {
        return {a * b.x, a * b.y, a * b.z};
    }

    __device__ __host__ inline vec3 operator/(float a, const vec3 &b) {
        return {a / b.x, a / b.y, a / b.z};
    }
}
#endif //ENGINE24_VEC3_CUH
