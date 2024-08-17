//
// Created by alex on 17.08.24.
//

#include "vec2.cuh"
#include "vec3.cuh"
#include "vec4.cuh"
#include "mat3.cuh"

namespace Bcg {
    __device__ __host__ vec3::vec3() : x(0), y(0), z(0), w(0) {

    }

    __device__ __host__ vec3::vec3(const vec2 &v) : x(v.x), y(v.y), z(0), w(0) {

    }

    __device__ __host__ vec3::vec3(float x, float y, float z) : x(x), y(y), z(z), w(0) {

    }

    __device__ __host__ vec3 vec3::constant(float c) {
        return {c, c, c};
    }

    __device__ __host__ vec3 vec3::unit(int i) {
        return {float(i == 0), float(i == 1), float(i == 2)};
    }

    __device__ __host__ vec3 vec3::operator-() const {
        return {-x, -y, -z};
    }

    __device__ __host__ float vec3::operator[](int i) const {
        return (&x)[i];
    }

    __device__ __host__ float &vec3::operator[](int i) {
        return (&x)[i];
    }

    __device__ __host__ float vec3::dot(const vec3 &b) const {
        return x * b.x + y * b.y + z * b.z;
    }

    __device__ __host__ float vec3::length2() const {
        return dot(*this);
    }

    __device__ __host__ float vec3::length() const {
        return sqrtf(length2());
    }

    __device__ __host__ vec3 vec3::normalized() const {
        return {*this / length()};
    }

    __device__ __host__ vec3 vec3::cross(const vec3 &b) const {
        return {y * b.z - z * b.y,
                z * b.x - x * b.z,
                x * b.y - y * b.x};
    }

    __device__ __host__ mat3 vec3::wedge() const {
        return mat3{
                {0,  -z, y},
                {z,  0,  -x},
                {-y, x,  0}
        };
    }

    __device__ __host__ mat3 vec3::outer(const vec3 &b) const {
        return mat3{
                {x * b.x, x * b.y, x * b.z},
                {y * b.x, y * b.y, y * b.z},
                {z * b.x, z * b.y, z * b.z}
        };
    }

    __device__ __host__ mat3 vec3::as_diag() const {
        return mat3{
                {x, 0, 0},
                {0, y, 0},
                {0, 0, z}
        };
    }

    __device__ __host__ vec3::operator vec2() const {
        return vec2{x, y};
    }

    __device__ __host__ vec3::operator vec4() const {
        return vec4{x, y, z, 1.0};
    }

    __device__ __host__ vec4 vec3::homogeneous() const {
        return vec4{x, y, z, 1.0};
    }

    __device__ __host__ vec3 vec3::operator+(const vec3 &b) const {
        return {x + b.x, y + b.y, z + b.z};
    }

    __device__ __host__ vec3 vec3::operator-(const vec3 &b) const {
        return {x - b.x, y - b.y, z - b.z};
    }

    __device__ __host__ vec3 vec3::operator*(const vec3 &b) const {
        return {x * b.x, y * b.y, z * b.z};
    }

    __device__ __host__ vec3 vec3::operator/(const vec3 &b) const {
        return {x / b.x, y / b.y, z / b.z};
    }

    __device__ __host__ vec3 vec3::operator+(const float &b) const {
        return {x + b, y + b, z + b};
    }

    __device__ __host__ vec3 vec3::operator-(const float &b) const {
        return {x - b, y - b, z - b};
    }

    __device__ __host__ vec3 vec3::operator*(const float &b) const {
        return {x * b, y * b, z * b};
    }

    __device__ __host__ vec3 vec3::operator/(const float &b) const {
        return {x / b, y / b, z / b};
    }


    __device__ __host__ vec3 operator+(const float &a, const vec3 &b) {
        return {a + b.x, a + b.y, a + b.z};
    }

    __device__ __host__ vec3 operator-(const float &a, const vec3 &b) {
        return {a - b.x, a - b.y, a - b.z};
    }

    __device__ __host__ vec3 operator*(const float &a, const vec3 &b) {
        return {a * b.x, a * b.y, a * b.z};
    }

    __device__ __host__ vec3 operator/(const float &a, const vec3 &b) {
        return {a / b.x, a / b.y, a / b.z};
    }
}