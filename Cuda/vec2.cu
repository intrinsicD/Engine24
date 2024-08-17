//
// Created by alex on 17.08.24.
//

#include "vec2.cuh"
#include "vec3.cuh"
#include "mat2.cuh"

namespace Bcg {
    __device__ __host__ vec2::vec2() : x(0), y(0) {

    }

    __device__ __host__ vec2::vec2(float x, float y) : x(x), y(y) {

    }

    __device__ __host__  vec2 vec2::constant(float c) {
        return {c, c};
    }

    __device__ __host__  vec2 vec2::unit(int i) {
        return {float(i == 0), float(i == 1)};
    }

    __device__ __host__ vec2 vec2::operator-() const {
        return {-x, -y};
    }

    __device__ __host__  float vec2::operator[](int i) const {
        return (&x)[i];
    }

    __device__ __host__  float &vec2::operator[](int i) {
        return (&x)[i];
    }

    __device__ __host__  float vec2::dot(const vec2 &b) const {
        return x * b.x + y * b.y;
    }

    __device__ __host__  float vec2::length2() const {
        return dot(*this);
    }

    __device__ __host__  float vec2::length() const {
        return sqrtf(length2());
    }

    __device__ __host__ vec2 vec2::normalized() const {
        return {*this / length()};
    }

    __device__ __host__  float vec2::cross(const vec2 &b) const {
        return x * b.y - y * b.x;
    }

    __device__ __host__  vec2::operator vec3() const {
        return vec3{x, y, 1.0};
    }

    __device__ __host__  vec3 vec2::homogeneous() const {
        return vec3{x, y, 1.0};
    }

    __device__ __host__ vec2 vec2::operator+(const vec2 &b) const {
        return {x + b.x, y + b.y};
    }

    __device__ __host__ vec2 vec2::operator-(const vec2 &b) const {
        return {x - b.x, y - b.y};
    }

    __device__ __host__ vec2 vec2::operator*(const vec2 &b) const {
        return {x * b.x, y * b.y};
    }

    __device__ __host__ vec2 vec2::operator/(const vec2 &b) const {
        return {x / b.x, y / b.y};
    }

    __device__ __host__ vec2 vec2::operator+(const float &b) const {
        return {x + b, y + b};
    }

    __device__ __host__ vec2 vec2::operator-(const float &b) const {
        return {x - b, y - b};
    }

    __device__ __host__ vec2 vec2::operator*(const float &b) const {
        return {x * b, y * b};
    }

    __device__ __host__ vec2 vec2::operator/(const float &b) const {
        return {x / b, y / b};
    }

    __device__ __host__  mat2 vec2::outer(const vec2 &b) const {
        return mat2{
                {x * b.x, x * b.y},
                {y * b.x, y * b.y}
        };
    }

    __device__ __host__  mat2 vec2::as_diag() const {
        return mat2{
                {x, 0},
                {0, y}
        };
    }


    __device__ __host__  vec2 operator+(const float &a, const vec2 &b) {
        return {a + b.x, a + b.y};
    }

    __device__ __host__  vec2 operator-(const float &a, const vec2 &b) {
        return {a - b.x, a - b.y};
    }

    __device__ __host__  vec2 operator*(const float &a, const vec2 &b) {
        return {a * b.x, a * b.y};
    }

    __device__ __host__  vec2 operator/(const float &a, const vec2 &b) {
        return {a / b.x, a / b.y};
    }
}