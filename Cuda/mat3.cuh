//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_MAT3_CUH
#define ENGINE24_MAT3_CUH

#include "vec3.cuh"

namespace Bcg {
    struct mat2;

    struct mat3 {
        vec3 col0, col1, col2;

        __device__ __host__ mat3();

        __device__ __host__ mat3(vec3 col0, vec3 col1, vec3 col2);

        __device__ __host__ static mat3 identity();

        __device__ __host__ static mat3 constant(float c);

        __device__ __host__ static mat3 reflect_x();

        __device__ __host__ static mat3 reflect_y();

        __device__ __host__ static mat3 reflect_z();

        __device__ __host__ static mat3 rot(const vec3 axis, float angle);

        __device__ __host__ static mat3 scale(const vec3 s);

        __device__ __host__ static mat3 project(const vec3 &normal);

        __device__ __host__ static mat3 shear_x(float s);

        __device__ __host__ static mat3 shear_y(float s);

        __device__ __host__ static mat3 shear_z(float s);

        __device__ __host__ mat2 left_upper() const;

        __device__ __host__ mat2 right_upper() const;

        __device__ __host__ mat2 left_lower() const;

        __device__ __host__ mat2 right_lower() const;

        __device__ __host__ mat3 operator-() const;

        __device__ __host__ vec3 operator[](int i) const;

        __device__ __host__ vec3 &operator[](int i);

        __device__ __host__ const float &operator()(int r, int c) const;

        __device__ __host__ float &operator()(int r, int c);

        __device__ __host__ mat3 operator+(const mat3 &b) const;

        __device__ __host__ mat3 operator-(const mat3 &b) const;

        __device__ __host__ mat3 operator*(const mat3 &b) const;

        __device__ __host__ mat3 operator+(float b) const;

        __device__ __host__ mat3 operator-(float b) const;

        __device__ __host__ mat3 operator*(float b) const;

        __device__ __host__ mat3 operator/(float b) const;

        __device__ __host__ vec3 operator*(const vec3 &v) const;

        __device__ __host__ mat3 transpose() const;

        __device__ __host__ float determinant() const;

        __device__ __host__ mat3 inverse() const;

        __device__ __host__ mat3 adjoint() const;

        __device__ __host__ mat3 cofactor() const;
    };

    __device__ __host__ mat3 operator+(float a, const mat3 &b);

    __device__ __host__ mat3 operator-(float a, const mat3 &b);

    __device__ __host__ mat3 operator*(float a, const mat3 &b);

    __device__ __host__ mat3 operator/(float a, const mat3 &b);

    __device__ __host__  float mat3_determinant(
            float a00, float a01, float a02,
            float a10, float a11, float a12,
            float a20, float a21, float a22);
}

#endif //ENGINE24_MAT3_CUH
