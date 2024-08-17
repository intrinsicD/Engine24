//
// Created by alex on 17.08.24.
//

#ifndef ENGINE24_VEC3_CUH
#define ENGINE24_VEC3_CUH

namespace Bcg {
    struct vec2;
    struct vec4;
    struct mat3;

    struct vec3 {
        float x, y, z, w;

        __device__ __host__ vec3();

        __device__ __host__ vec3(const vec2 &v);

        __device__ __host__ vec3(float x, float y, float z);

        __device__ __host__ static vec3 constant(float c);

        __device__ __host__ static vec3 unit(int i);

        __device__ __host__ vec3 operator-() const;

        __device__ __host__ float operator[](int i) const;

        __device__ __host__ float &operator[](int i);

        __device__ __host__ float dot(const vec3 &b) const;

        __device__ __host__ float length2() const;

        __device__ __host__ float length() const;

        __device__ __host__ vec3 normalized() const;

        __device__ __host__ vec3 cross(const vec3 &b) const;

        __device__ __host__ operator vec2() const;

        __device__ __host__ operator vec4() const;

        __device__ __host__ vec4 homogeneous() const;

        __device__ __host__ vec3 operator+(const vec3 &b) const;

        __device__ __host__ vec3 operator-(const vec3 &b) const;

        __device__ __host__ vec3 operator*(const vec3 &b) const;

        __device__ __host__ vec3 operator/(const vec3 &b) const;

        __device__ __host__ vec3 operator+(const float &b) const;

        __device__ __host__ vec3 operator-(const float &b) const;

        __device__ __host__ vec3 operator*(const float &b) const;

        __device__ __host__ vec3 operator/(const float &b) const;

        __device__ __host__ mat3 wedge() const;

        __device__ __host__ mat3 outer(const vec3 &b) const;

        __device__ __host__ mat3 as_diag() const;
    };

    __device__ __host__ vec3 operator+(const float &a, const vec3 &b);

    __device__ __host__ vec3 operator-(const float &a, const vec3 &b);

    __device__ __host__ vec3 operator*(const float &a, const vec3 &b);

    __device__ __host__ vec3 operator/(const float &a, const vec3 &b);
}
#endif //ENGINE24_VEC3_CUH
