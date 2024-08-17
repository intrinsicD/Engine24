//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_VEC2_CUH
#define ENGINE24_VEC2_CUH

namespace Bcg {
    struct mat2;
    struct vec3;

    struct vec2 {
        float x, y;

        __device__ __host__ vec2();

        __device__ __host__ vec2(float x, float y);

        __device__ __host__ static vec2 constant(float c);

        __device__ __host__ static vec2 unit(int i);

        __device__ __host__ vec2 operator-() const;

        __device__ __host__ float operator[](int i) const;

        __device__ __host__ float &operator[](int i);

        __device__ __host__ float dot(const vec2 &b) const;

        __device__ __host__ float length2() const;

        __device__ __host__ float length() const;

        __device__ __host__ vec2 normalized() const;

        __device__ __host__ float cross(const vec2 &b) const;

        __device__ __host__ operator vec3() const;

        __device__ __host__ vec3 homogeneous() const;

        __device__ __host__ vec2 operator+(const vec2 &b) const;

        __device__ __host__ vec2 operator-(const vec2 &b) const;

        __device__ __host__ vec2 operator*(const vec2 &b) const;

        __device__ __host__ vec2 operator/(const vec2 &b) const;

        __device__ __host__ vec2 operator+(const float &b) const;

        __device__ __host__ vec2 operator-(const float &b) const;

        __device__ __host__ vec2 operator*(const float &b) const;

        __device__ __host__ vec2 operator/(const float &b) const;

        __device__ __host__ mat2 outer(const vec2 &b) const;

        __device__ __host__ mat2 as_diag() const;
    };

    __device__ __host__ vec2 operator+(const float &a, const vec2 &b);

    __device__ __host__ vec2 operator-(const float &a, const vec2 &b);

    __device__ __host__ vec2 operator*(const float &a, const vec2 &b);

    __device__ __host__ vec2 operator/(const float &a, const vec2 &b);
}
#endif //ENGINE24_VEC2_CUH
