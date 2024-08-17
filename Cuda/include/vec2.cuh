//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_VEC2_CUH
#define ENGINE24_VEC2_CUH

namespace Bcg::cuda {
    struct vec2 {
        float x, y;

        __device__ __host__ inline vec2() : x(0), y(0) {

        }

        __device__ __host__ inline vec2(float x, float y) : x(x), y(y) {

        }

        __device__ __host__ inline static vec2 constant(float c);

        __device__ __host__ inline static vec2 unit(int i);

        __device__ __host__ inline vec2 operator-() const;

        __device__ __host__ inline float operator[](int i) const;

        __device__ __host__ inline float &operator[](int i);

        __device__ __host__ inline float dot(const vec2 &b) const;

        __device__ __host__ inline float length2() const;

        __device__ __host__ inline float length() const;

        __device__ __host__ inline vec2 normalized() const;

        __device__ __host__ inline float cross(const vec2 &b) const;

        __device__ __host__ inline vec2 operator+(const vec2 &b) const;

        __device__ __host__ inline vec2 operator-(const vec2 &b) const;

        __device__ __host__ inline vec2 operator*(const vec2 &b) const;

        __device__ __host__ inline vec2 operator/(const vec2 &b) const;

        __device__ __host__ inline vec2 operator+(float b) const;

        __device__ __host__ inline vec2 operator-(float b) const;

        __device__ __host__ inline vec2 operator*(float b) const;

        __device__ __host__ inline vec2 operator/(float b) const;
    };

    __device__ __host__ inline vec2 vec2::constant(float c) {
        return {c, c};
    }

    __device__ __host__ inline vec2 vec2::unit(int i) {
        return {float(i == 0), float(i == 1)};
    }

    __device__ __host__ inline vec2 vec2::operator-() const {
        return {-x, -y};
    }

    __device__ __host__ inline float vec2::operator[](int i) const {
        return (&x)[i];
    }

    __device__ __host__ inline float &vec2::operator[](int i) {
        return (&x)[i];
    }

    __device__ __host__ inline float vec2::dot(const vec2 &b) const {
        return x * b.x + y * b.y;
    }

    __device__ __host__ inline float vec2::length2() const {
        return dot(*this);
    }

    __device__ __host__ inline float vec2::length() const {
        return sqrtf(length2());
    }

    __device__ __host__ inline vec2 vec2::normalized() const {
        return {*this / length()};
    }

    __device__ __host__ inline float vec2::cross(const vec2 &b) const {
        return x * b.y - y * b.x;
    }

    __device__ __host__ inline vec2 vec2::operator+(const vec2 &b) const {
        return {x + b.x, y + b.y};
    }

    __device__ __host__ inline vec2 vec2::operator-(const vec2 &b) const {
        return {x - b.x, y - b.y};
    }

    __device__ __host__ inline vec2 vec2::operator*(const vec2 &b) const {
        return {x * b.x, y * b.y};
    }

    __device__ __host__ inline vec2 vec2::operator/(const vec2 &b) const {
        return {x / b.x, y / b.y};
    }

    __device__ __host__ inline vec2 vec2::operator+(float b) const {
        return {x + b, y + b};
    }

    __device__ __host__ inline vec2 vec2::operator-(float b) const {
        return {x - b, y - b};
    }

    __device__ __host__ inline vec2 vec2::operator*(float b) const {
        return {x * b, y * b};
    }

    __device__ __host__ inline vec2 vec2::operator/(float b) const {
        return {x / b, y / b};
    }

    __device__ __host__ inline vec2 operator+(float a, const vec2 &b) {
        return {a + b.x, a + b.y};
    }

    __device__ __host__ inline vec2 operator-(float a, const vec2 &b) {
        return {a - b.x, a - b.y};
    }

    __device__ __host__ inline vec2 operator*(float a, const vec2 &b) {
        return {a * b.x, a * b.y};
    }

    __device__ __host__ inline vec2 operator/(float a, const vec2 &b) {
        return {a / b.x, a / b.y};
    }
}
#endif //ENGINE24_VEC2_CUH
