//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_VEC4_CUH
#define ENGINE24_VEC4_CUH

namespace Bcg {
    struct vec2;
    struct vec3;
    struct mat4;

    struct vec4 {
        float x, y, z, w;

        __device__ __host__ vec4();

        __device__ __host__ vec4(float x);

        __device__ __host__ vec4(float *x);

        __device__ __host__ vec4(const vec2 &v);

        __device__ __host__ vec4(const vec3 &v);

        __device__ __host__ vec4(float x, float y, float z, float w);

        __device__ __host__ vec4 operator-() const;

        __device__ __host__ float operator[](int i) const;

        __device__ __host__ float &operator[](int i);

        __device__ __host__ float dot(const vec4 &b) const;

        __device__ __host__ float length2() const;

        __device__ __host__ float length() const;

        __device__ __host__ vec4 normalized() const;

        __device__ __host__ operator vec3() const;

        __device__ __host__ vec4 operator+(const vec4 &b) const;

        __device__ __host__ vec4 operator-(const vec4 &b) const;

        __device__ __host__ vec4 operator*(const vec4 &b) const;

        __device__ __host__ vec4 operator/(const vec4 &b) const;

        __device__ __host__ vec4 operator+(const float &b) const;

        __device__ __host__ vec4 operator-(const float &b) const;

        __device__ __host__ vec4 operator*(const float &b) const;

        __device__ __host__ vec4 operator/(const float &b) const;

        __device__ __host__ mat4 outer(const vec4 &b) const;

        __device__ __host__ mat4 as_diag() const;
    };

    __device__ __host__  vec4 operator+(const float &a, const vec4 &b);

    __device__ __host__  vec4 operator-(const float &a, const vec4 &b);

    __device__ __host__  vec4 operator*(const float &a, const vec4 &b);

    __device__ __host__  vec4 operator/(const float &a, const vec4 &b);
}

#endif //ENGINE24_VEC4_CUH
