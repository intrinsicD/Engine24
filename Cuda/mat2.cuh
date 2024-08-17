//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_MAT2_CUH
#define ENGINE24_MAT2_CUH

#include "vec2.cuh"

namespace Bcg {
    struct mat2 {
        vec2 col0, col1;

        __device__ __host__ mat2();

        __device__ __host__ mat2(vec2 col0, vec2 col1);

        __device__ __host__ static mat2 identity();

        __device__ __host__ static mat2 constant(float c);

        __device__ __host__ static mat2 reflect_x();

        __device__ __host__ static mat2 reflect_y();

        __device__ __host__ static mat2 rot(float angle);

        __device__ __host__ static mat2 project(float angle);

        __device__ __host__ static mat2 shear_x(float s);

        __device__ __host__ static mat2 shear_y(float s);

        __device__ __host__ mat2 operator-() const;

        __device__ __host__ vec2 operator[](int i) const;

        __device__ __host__ vec2 &operator[](int i);

        __device__ __host__ const float &operator()(int r, int c) const;

        __device__ __host__ float &operator()(int r, int c);

        __device__ __host__ mat2 operator+(const mat2 &b) const;

        __device__ __host__ mat2 operator-(const mat2 &b) const;

        __device__ __host__ mat2 operator*(const mat2 &b) const;

        __device__ __host__ mat2 operator+(float b) const;

        __device__ __host__ mat2 operator-(float b) const;

        __device__ __host__ mat2 operator*(float b) const;

        __device__ __host__ mat2 operator/(float b) const;

        __device__ __host__ vec2 operator*(const vec2 &v) const;

        __device__ __host__ mat2 transpose() const;

        __device__ __host__ float determinant() const;

        __device__ __host__ mat2 inverse() const;

        __device__ __host__ mat2 adjoint() const;

        __device__ __host__ mat2 cofactor() const;
    };

    __device__ __host__ mat2 operator+(float a, const mat2 &b);

    __device__ __host__ mat2 operator-(float a, const mat2 &b);

    __device__ __host__ mat2 operator*(float a, const mat2 &b);

    __device__ __host__ mat2 operator/(float a, const mat2 &b);

    __device__ __host__ float mat2_determinant(float a, float b, float c, float d);

}

#endif //ENGINE24_MAT2_CUH
