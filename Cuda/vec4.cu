//
// Created by alex on 17.08.24.
//
#include "vec4.cuh"
#include "vec2.cuh"
#include "vec3.cuh"
#include "mat4.cuh"

namespace Bcg {
    __device__ __host__  vec4::vec4() : x(0), y(0), z(0), w(0) {

    }

    __device__ __host__  vec4::vec4(float x) : x(x), y(x), z(x), w(x) {

    }

    __device__ __host__  vec4::vec4(float *x) : x(x[0]), y(x[1]), z(x[2]), w(x[3]) {

    }

    __device__ __host__  vec4::vec4(const vec2 &v) : x(v.x), y(v.y), z(0), w(0) {

    }

    __device__ __host__  vec4::vec4(const vec3 &v) : x(v.x), y(v.y), z(v.z), w(0) {

    }

    __device__ __host__  vec4::vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {

    }

    __device__ __host__  vec4 vec4::operator-() const {
        return {-x, -y, -z, -w};
    }

    __device__ __host__  float vec4::operator[](int i) const {
        return (&x)[i];
    }

    __device__ __host__  float &vec4::operator[](int i) {
        return (&x)[i];
    }

    __device__ __host__  float vec4::dot(const vec4 &b) const {
        return x * b.x + y * b.y + z * b.z + w * b.w;
    }

    __device__ __host__  float vec4::length2() const {
        return dot(*this);
    }

    __device__ __host__  float vec4::length() const {
        return sqrtf(length2());
    }

    __device__ __host__  vec4 vec4::normalized() const {
        return {*this / length()};
    }

    __device__ __host__  vec4::operator vec3() const {
        return vec3{x, y, z};
    }

    __device__ __host__ mat4 vec4::outer(const vec4 &b) const {
        return mat4{
                {x * b.x, x * b.y, x * b.z, x * b.w},
                {y * b.x, y * b.y, y * b.z, y * b.w},
                {z * b.x, z * b.y, z * b.z, z * b.w},
                {w * b.x, w * b.y, w * b.z, w * b.w}
        };
    }

    __device__ __host__ mat4 vec4::as_diag() const {
        return mat4{
                {x, 0, 0, 0},
                {0, y, 0, 0},
                {0, 0, z, 0},
                {0, 0, 0, w}
        };
    }

    __device__ __host__  vec4 vec4::operator+(const vec4 &b) const {
        return {x + b.x, y + b.y, z + b.z, w + b.w};
    }

    __device__ __host__  vec4 vec4::operator-(const vec4 &b) const {
        return {x - b.x, y - b.y, z - b.z, w - b.w};
    }

    __device__ __host__  vec4 vec4::operator*(const vec4 &b) const {
        return {x * b.x, y * b.y, z * b.z, w * b.w};
    }

    __device__ __host__  vec4 vec4::operator/(const vec4 &b) const {
        return {x / b.x, y / b.y, z / b.z, w / b.w};
    }

    __device__ __host__  vec4 vec4::operator+(const float &b) const {
        return {x + b, y + b, z + b, w + b};
    }

    __device__ __host__  vec4 vec4::operator-(const float &b) const {
        return {x - b, y - b, z - b, w - b};
    }

    __device__ __host__  vec4 vec4::operator*(const float &b) const {
        return {x * b, y * b, z * b, w * b};
    }

    __device__ __host__  vec4 vec4::operator/(const float &b) const {
        return {x / b, y / b, z / b, w / b};
    }


    __device__ __host__  vec4 operator+(const float &a, const vec4 &b) {
        return {a + b.x, a + b.y, a + b.z, a + b.w};
    }

    __device__ __host__  vec4 operator-(const float &a, const vec4 &b) {
        return {a - b.x, a - b.y, a - b.z, a - b.w};
    }

    __device__ __host__  vec4 operator*(const float &a, const vec4 &b) {
        return {a * b.x, a * b.y, a * b.z, a * b.w};
    }

    __device__ __host__  vec4 operator/(const float &a, const vec4 &b) {
        return {a / b.x, a / b.y, a / b.z, a / b.w};
    }
}