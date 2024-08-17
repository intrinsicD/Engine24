//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_MAT4_CUH
#define ENGINE24_MAT4_CUH

#include "vec4.cuh"

namespace Bcg {
    struct mat3;

    struct mat4 {
        vec4 col0, col1, col2, col3;

        __device__ __host__ mat4();

        __device__ __host__ mat4(vec4 col0, vec4 col1, vec4 col2, vec4 col3);

        __device__ __host__ mat4(const mat3 &upper_left);

        __device__ __host__ operator mat3() const;

        __device__ __host__ static mat4 identity();

        __device__ __host__ static mat4 constant(float c);

        __device__ __host__ static mat4 reflect_x();

        __device__ __host__ static mat4 reflect_y();

        __device__ __host__ static mat4 reflect_z();

        __device__ __host__ static mat4 reflect_w();

        __device__ __host__ static mat4 rot(const vec3 axis, float angle);

        __device__ __host__ static mat4 translation(const vec3 t);

        __device__ __host__ static mat4 scale(const vec3 s);

        __device__ __host__ static mat4 shear_x(float s);

        __device__ __host__ static mat4 shear_y(float s);

        __device__ __host__ static mat4 shear_z(float s);

        __device__ __host__ static mat4 shear_w(float s);

        __device__ __host__ mat3 left_upper() const;

        __device__ __host__ mat3 right_upper() const;

        __device__ __host__ mat3 left_lower() const;

        __device__ __host__ mat3 right_lower() const;

        __device__ __host__ mat4 operator-() const;

        __device__ __host__ vec4 operator[](int i) const;

        __device__ __host__ vec4 &operator[](int i);

        __device__ __host__ const float &operator()(int r, int c) const;

        __device__ __host__ float &operator()(int r, int c);

        __device__ __host__ mat4 operator+(const mat4 &b) const;

        __device__ __host__ mat4 operator-(const mat4 &b) const;

        __device__ __host__ mat4 operator*(const mat4 &b) const;

        __device__ __host__ mat4 operator+(float b) const;

        __device__ __host__ mat4 operator-(float b) const;

        __device__ __host__ mat4 operator*(float b) const;

        __device__ __host__ mat4 operator/(float b) const;

        __device__ __host__ vec4 operator*(const vec4 &v) const;

        __device__ __host__ mat4 transpose() const;

        __device__ __host__ float determinant() const;

        __device__ __host__ mat4 inverse() const;

        __device__ __host__ mat4 adjoint() const;

        __device__ __host__ mat4 cofactor() const;
    };

    __device__ __host__ float mat4_determinant(float a00, float a01, float a02, float a03,
                                               float a10, float a11, float a12, float a13,
                                               float a20, float a21, float a22, float a23,
                                               float a30, float a31, float a32, float a33);
}

#endif //ENGINE24_MAT4_CUH
