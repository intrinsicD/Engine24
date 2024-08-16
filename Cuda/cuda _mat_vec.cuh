//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_CUDA__MAT_VEC_CUH
#define ENGINE24_CUDA__MAT_VEC_CUH

namespace Bcg {
    struct mat2;
    struct mat3;
    struct mat4;

    struct vec2 {
        union {
            struct {
                float x, y;
            };
            float data[2];
        };

        __device__ __host__ inline vec2() : x(0), y(0) {

        }

        __device__ __host__ inline vec2(float x, float y) : x(x), y(y) {

        }

        __device__ __host__ inline vec2 operator-() const {
            return {-x, -y};
        }

        __device__ __host__ inline float operator[](int i) const {
            return data[i];
        }

        __device__ __host__ inline float &operator[](int i) {
            return data[i];
        }

        __device__ __host__ inline float dot(const vec2 &b) const {
            return x * b.x + y * b.y;
        }

        __device__ __host__ inline float length2() const {
            return dot(*this);
        }

        __device__ __host__ inline float length() const {
            return sqrtf(length2());
        }

        __device__ __host__ inline vec2 normalized() const {
            return {*this / length()};
        }

        __device__ __host__ inline float cross(const vec2 &b) const {
            return x * b.y - y * b.x;
        }

        __device__ __host__ inline vec2 operator+(const vec2 &b) const {
            return {x + b.x, y + b.y};
        }

        __device__ __host__ inline vec2 operator-(const vec2 &b) const {
            return {x - b.x, y - b.y};
        }

        __device__ __host__ inline vec2 operator*(const vec2 &b) const {
            return {x * b.x, y * b.y};
        }

        __device__ __host__ inline vec2 operator/(const vec2 &b) const {
            return {x / b.x, y / b.y};
        }

        __device__ __host__ inline vec2 operator+(const float &b) const {
            return {x + b, y + b};
        }

        __device__ __host__ inline vec2 operator-(const float &b) const {
            return {x - b, y - b};
        }

        __device__ __host__ inline vec2 operator*(const float &b) const {
            return {x * b, y * b};
        }

        __device__ __host__ inline vec2 operator/(const float &b) const {
            return {x / b, y / b};
        }

        __device__ __host__ inline mat2 outer(const vec2 &b) const;
    };

    __device__ __host__ inline vec2 operator+(const float &a, const vec2 &b) {
        return {a + b.x, a + b.y};
    }

    __device__ __host__ inline vec2 operator-(const float &a, const vec2 &b) {
        return {a - b.x, a - b.y};
    }

    __device__ __host__ inline vec2 operator*(const float &a, const vec2 &b) {
        return {a * b.x, a * b.y};
    }

    __device__ __host__ inline vec2 operator/(const float &a, const vec2 &b) {
        return {a / b.x, a / b.y};
    }

    //------------------------------------------------------------------------------------------------------------------
    // Vector vec3
    //------------------------------------------------------------------------------------------------------------------

    struct vec3 {
        union {
            struct {
                float x, y, z, w;
            };
            float data[4];
        };

        __device__ __host__ inline vec3() : x(0), y(0), z(0), w(0) {

        }

        __device__ __host__ inline vec3(float x, float y, float z) : x(x), y(y), z(z), w(0) {

        }

        __device__ __host__ inline vec3 operator-() const {
            return {-x, -y, -z};
        }

        __device__ __host__ inline float operator[](int i) const {
            return data[i];
        }

        __device__ __host__ inline float &operator[](int i) {
            return data[i];
        }

        __device__ __host__ inline float dot(const vec3 &b) const {
            return x * b.x + y * b.y + z * b.z;
        }

        __device__ __host__ inline float length2() const {
            return dot(*this);
        }

        __device__ __host__ inline float length() const {
            return sqrtf(length2());
        }

        __device__ __host__ inline vec3 normalized() const {
            return {*this / length()};
        }

        __device__ __host__ inline vec3 cross(const vec3 &b) const {
            return {y * b.z - z * b.y,
                    z * b.x - x * b.z,
                    x * b.y - y * b.x};
        }

        __device__ __host__ inline mat3 wedge() const;

        __device__ __host__ inline mat3 outer(const vec3 &b) const;

        __device__ __host__ inline vec3 operator+(const vec3 &b) const {
            return {x + b.x, y + b.y, z + b.z};
        }

        __device__ __host__ inline vec3 operator-(const vec3 &b) const {
            return {x - b.x, y - b.y, z - b.z};
        }

        __device__ __host__ inline vec3 operator*(const vec3 &b) const {
            return {x * b.x, y * b.y, z * b.z};
        }

        __device__ __host__ inline vec3 operator/(const vec3 &b) const {
            return {x / b.x, y / b.y, z / b.z};
        }

        __device__ __host__ inline vec3 operator+(const float &b) const {
            return {x + b, y + b, z + b};
        }

        __device__ __host__ inline vec3 operator-(const float &b) const {
            return {x - b, y - b, z - b};
        }

        __device__ __host__ inline vec3 operator*(const float &b) const {
            return {x * b, y * b, z * b};
        }

        __device__ __host__ inline vec3 operator/(const float &b) const {
            return {x / b, y / b, z / b};
        }
    };

    __device__ __host__ inline vec3 operator+(const float &a, const vec3 &b) {
        return {a + b.x, a + b.y, a + b.z};
    }

    __device__ __host__ inline vec3 operator-(const float &a, const vec3 &b) {
        return {a - b.x, a - b.y, a - b.z};
    }

    __device__ __host__ inline vec3 operator*(const float &a, const vec3 &b) {
        return {a * b.x, a * b.y, a * b.z};
    }

    __device__ __host__ inline vec3 operator/(const float &a, const vec3 &b) {
        return {a / b.x, a / b.y, a / b.z};
    }

    //------------------------------------------------------------------------------------------------------------------
    // Vector vec4
    //------------------------------------------------------------------------------------------------------------------

    struct vec4 {
        union {
            struct {
                float x, y, z, w;
            };
            float data[4];
        };

        __device__ __host__ inline vec4() : x(0), y(0), z(0), w(0) {

        }

        __device__ __host__ inline vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {

        }

        __device__ __host__ inline vec4 operator-() const {
            return {-x, -y, -z, -w};
        }

        __device__ __host__ inline float operator[](int i) const {
            return data[i];
        }

        __device__ __host__ inline float &operator[](int i) {
            return data[i];
        }

        __device__ __host__ inline float dot(const vec4 &b) const {
            return x * b.x + y * b.y + z * b.z + w * b.w;
        }

        __device__ __host__ inline float length2() const {
            return dot(*this);
        }

        __device__ __host__ inline float length() const {
            return sqrtf(length2());
        }

        __device__ __host__ inline vec4 normalized() const {
            return {*this / length()};
        }

        __device__ __host__ inline mat4 wedge() const;

        __device__ __host__ inline mat4 outer(const vec4 &b) const;

        __device__ __host__ inline vec4 operator+(const vec4 &b) const {
            return {x + b.x, y + b.y, z + b.z, w + b.w};
        }

        __device__ __host__ inline vec4 operator-(const vec4 &b) const {
            return {x - b.x, y - b.y, z - b.z, w - b.w};
        }

        __device__ __host__ inline vec4 operator*(const vec4 &b) const {
            return {x * b.x, y * b.y, z * b.z, w * b.w};
        }

        __device__ __host__ inline vec4 operator/(const vec4 &b) const {
            return {x / b.x, y / b.y, z / b.z, w / b.w};
        }

        __device__ __host__ inline vec4 operator+(const float &b) const {
            return {x + b, y + b, z + b, w + b};
        }

        __device__ __host__ inline vec4 operator-(const float &b) const {
            return {x - b, y - b, z - b, w - b};
        }

        __device__ __host__ inline vec4 operator*(const float &b) const {
            return {x * b, y * b, z * b, w * b};
        }

        __device__ __host__ inline vec4 operator/(const float &b) const {
            return {x / b, y / b, z / b, w / b};
        }

    };

    __device__ __host__ inline vec4 operator+(const float &a, const vec4 &b) {
        return {a + b.x, a + b.y, a + b.z, a + b.w};
    }

    __device__ __host__ inline vec4 operator-(const float &a, const vec4 &b) {
        return {a - b.x, a - b.y, a - b.z, a - b.w};
    }

    __device__ __host__ inline vec4 operator*(const float &a, const vec4 &b) {
        return {a * b.x, a * b.y, a * b.z, a * b.w};
    }

    __device__ __host__ inline vec4 operator/(const float &a, const vec4 &b) {
        return {a / b.x, a / b.y, a / b.z, a / b.w};
    }

    //------------------------------------------------------------------------------------------------------------------
    // Matrix mat2
    //------------------------------------------------------------------------------------------------------------------

    __device__ __host__ inline float mat2_determinant(float a, float b, float c, float d) {
        return a * d - b * c;
    }

    struct mat2 {
        union {
            struct {
                vec2 x, y;
            };
            float data[4];
        };

        __device__ __host__ inline mat2() : x(), y() {

        }

        __device__ __host__ inline mat2(vec2 x, vec2 y) : x(x), y(y) {

        }

        __device__ __host__ inline mat2 operator-() const {
            return {-x, -y};
        }

        __device__ __host__ inline vec2 operator[](int i) const {
            return {data[2 * i], data[2 * i + 1]};
        }

        __device__ __host__ inline vec2 &operator[](int i) {
            return *reinterpret_cast<vec2 *>(data + 2 * i);
        }

        __device__ __host__ inline mat2 operator+(const mat2 &b) const {
            return {x + b.x, y + b.y};
        }

        __device__ __host__ inline mat2 operator-(const mat2 &b) const {
            return {x - b.x, y - b.y};
        }

        __device__ __host__ inline mat2 operator*(const mat2 &b) const {
            return {x * b.x, y * b.y};
        }

        __device__ __host__ inline mat2 operator+(const float &b) const {
            return {x + b, y + b};
        }

        __device__ __host__ inline mat2 operator-(const float &b) const {
            return {x - b, y - b};
        }

        __device__ __host__ inline mat2 operator*(const float &b) const {
            return {x * b, y * b};
        }

        __device__ __host__ inline mat2 operator/(const float &b) const {
            return {x / b, y / b};
        }

        __device__ __host__ inline vec2 operator*(const vec2 &v) const {
            return {x.x * v.x + x.y * v.y, y.x * v.x + y.y * v.y};
        }

        __device__ __host__ inline mat2 transpose() const {
            return {{x.x, y.x},
                    {x.y, y.y}};
        }

        __device__ __host__ inline float determinant() const {
            return mat2_determinant(x.x, x.y, y.x, y.y);
        }

        __device__ __host__ inline mat2 inverse() const {
            return transpose() / determinant();
        }

        __device__ __host__ inline mat2 adjoint() const {
            return {{y.y,  -x.y},
                    {-y.x, x.x}};
        }

        __device__ __host__ inline mat2 cofactor() const {
            return adjoint();
        }
    };

    __device__ __host__ inline mat2 operator+(const float &a, const mat2 &b) {
        return {a + b.x, a + b.y};
    }

    __device__ __host__ inline mat2 operator-(const float &a, const mat2 &b) {
        return {a - b.x, a - b.y};
    }

    __device__ __host__ inline mat2 operator*(const float &a, const mat2 &b) {
        return {a * b.x, a * b.y};
    }

    __device__ __host__ inline mat2 operator/(const float &a, const mat2 &b) {
        return {a / b.x, a / b.y};
    }

    __device__ __host__ inline mat2 vec2::outer(const vec2 &b) const {
        return mat2{
                {x * b.x, x * b.y},
                {y * b.x, y * b.y}
        };
    }

    //------------------------------------------------------------------------------------------------------------------
    // Matrix mat3
    //------------------------------------------------------------------------------------------------------------------

    __device__ __host__ inline float mat3_determinant(
            float a00, float a01, float a02,
            float a10, float a11, float a12,
            float a20, float a21, float a22) {
        return a00 * mat2_determinant(a11, a12, a21, a22)
               - a01 * mat2_determinant(a10, a12, a20, a22)
               + a02 * mat2_determinant(a10, a11, a20, a21);
    }

    struct mat3 {
        union {
            struct {
                vec3 x, y, z, w;
            };
            float data[16];
        };

        __device__ __host__ inline mat3() : x(), y(), z() {

        }

        __device__ __host__ inline mat3(vec3 x, vec3 y, vec3 z) : x(x), y(y), z(z) {

        }

        __device__ __host__ inline mat2 left_upper() const {
            return {{x.x, x.y},
                    {y.x, y.y}};
        }

        __device__ __host__ inline mat2 right_upper() const {
            return {{x.y, x.z},
                    {y.y, y.z}};
        }

        __device__ __host__ inline mat2 left_lower() const {
            return {{y.x, y.y},
                    {z.x, z.y}};
        }

        __device__ __host__ inline mat2 right_lower() const {
            return {{y.y, y.z},
                    {z.y, z.z}};
        }

        __device__ __host__ inline mat3 operator-() const {
            return {-x, -y, -z};
        }

        __device__ __host__ inline vec3 operator[](int i) const {
            return {data[3 * i], data[3 * i + 1], data[3 * i + 2]};
        }

        __device__ __host__ inline vec3 &operator[](int i) {
            return *reinterpret_cast<vec3 *>(data + 3 * i);
        }

        __device__ __host__ inline mat3 operator+(const mat3 &b) const {
            return {x + b.x, y + b.y, z + b.z};
        }

        __device__ __host__ inline mat3 operator-(const mat3 &b) const {
            return {x - b.x, y - b.y, z - b.z};
        }

        __device__ __host__ inline mat3 operator*(const mat3 &b) const {
            return {x * b.x, y * b.y, z * b.z};
        }

        __device__ __host__ inline mat3 operator+(const float &b) const {
            return {x + b, y + b, z + b};
        }

        __device__ __host__ inline mat3 operator-(const float &b) const {
            return {x - b, y - b, z - b};
        }

        __device__ __host__ inline mat3 operator*(const float &b) const {
            return {x * b, y * b, z * b};
        }

        __device__ __host__ inline mat3 operator/(const float &b) const {
            return {x / b, y / b, z / b};
        }

        __device__ __host__ inline vec3 operator*(const vec3 &v) const {
            return {x.x * v.x + x.y * v.y + x.z * v.z,
                    y.x * v.x + y.y * v.y + y.z * v.z,
                    z.x * v.x + z.y * v.y + z.z * v.z};
        }

        __device__ __host__ inline mat3 transpose() const {
            return {{x.x, y.x, z.x},
                    {x.y, y.y, z.y},
                    {x.z, y.z, z.z}};
        }

        __device__ __host__ inline float determinant() const {
            return mat3_determinant(x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z);
        }

        __device__ __host__ inline mat3 inverse() const {
            return transpose() / determinant();
        }

        __device__ __host__ inline mat3 adjoint() const {
            return mat3{
                    // First row of cofactors
                    vec3{
                            mat2_determinant(y.y, y.z, z.y, z.z),
                            -mat2_determinant(y.x, y.z, z.x, z.z),
                            mat2_determinant(y.x, y.y, z.x, z.y)
                    },
                    // Second row of cofactors
                    vec3{
                            -mat2_determinant(x.y, x.z, z.y, z.z),
                            mat2_determinant(x.x, x.z, z.x, z.z),
                            -mat2_determinant(x.x, x.y, z.x, z.y)
                    },
                    // Third row of cofactors
                    vec3{
                            mat2_determinant(x.y, x.z, y.y, y.z),
                            -mat2_determinant(x.x, x.z, y.x, y.z),
                            mat2_determinant(x.x, x.y, y.x, y.y)
                    }
            }.transpose();
        }

        __device__ __host__ inline mat3 cofactor() const {
            return adjoint();
        }
    };

    __device__ __host__ inline mat3 vec3::wedge() const {
        return mat3{
                {0,  -z, y},
                {z,  0,  -x},
                {-y, x,  0}
        };
    }

    __device__ __host__ inline mat3 vec3::outer(const vec3 &b) const {
        return mat3{
                {x * b.x, x * b.y, x * b.z},
                {y * b.x, y * b.y, y * b.z},
                {z * b.x, z * b.y, z * b.z}
        };
    }

    //------------------------------------------------------------------------------------------------------------------
    // Matrix mat4
    //------------------------------------------------------------------------------------------------------------------


    __device__ __host__ inline float mat4_determinant(float a00, float a01, float a02, float a03,
                                                      float a10, float a11, float a12, float a13,
                                                      float a20, float a21, float a22, float a23,
                                                      float a30, float a31, float a32, float a33) {
        return a00 * mat3_determinant(a11, a12, a13, a21, a22, a23, a31, a32, a33)
               - a01 * mat3_determinant(a10, a12, a13, a20, a22, a23, a30, a32, a33)
               + a02 * mat3_determinant(a10, a11, a13, a20, a21, a23, a30, a31, a33)
               - a03 * mat3_determinant(a10, a11, a12, a20, a21, a22, a30, a31, a32);
    }

    struct mat4 {
        union {
            struct {
                vec4 x, y, z, w;
            };
            float data[16];
        };

        __device__ __host__ inline mat4() : x(), y(), z(), w() {

        }

        __device__ __host__ inline mat4(vec4 x, vec4 y, vec4 z, vec4 w) : x(x), y(y), z(z), w(w) {

        }

        __device__ __host__ inline mat4 operator-() const {
            return {-x, -y, -z, -w};
        }

        __device__ __host__ inline vec4 operator[](int i) const {
            return {data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]};
        }

        __device__ __host__ inline vec4 &operator[](int i) {
            return *reinterpret_cast<vec4 *>(data + 4 * i);
        }

        __device__ __host__ inline mat4 operator+(const mat4 &b) const {
            return {x + b.x, y + b.y, z + b.z, w + b.w};
        }

        __device__ __host__ inline mat4 operator-(const mat4 &b) const {
            return {x - b.x, y - b.y, z - b.z, w - b.w};
        }

        __device__ __host__ inline mat4 operator*(const mat4 &b) const {
            return {x * b.x, y * b.y, z * b.z, w * b.w};
        }

        __device__ __host__ inline mat4 operator+(const float &b) const {
            return {x + b, y + b, z + b, w + b};
        }

        __device__ __host__ inline mat4 operator-(const float &b) const {
            return {x - b, y - b, z - b, w - b};
        }

        __device__ __host__ inline mat4 operator*(const float &b) const {
            return {x * b, y * b, z * b, w * b};
        }

        __device__ __host__ inline mat4 operator/(const float &b) const {
            return {x / b, y / b, z / b, w / b};
        }

        __device__ __host__ inline vec4 operator*(const vec4 &v) const {
            return {x.x * v.x + x.y * v.y + x.z * v.z + x.w * v.w,
                    y.x * v.x + y.y * v.y + y.z * v.z + y.w * v.w,
                    z.x * v.x + z.y * v.y + z.z * v.z + z.w * v.w,
                    w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w};
        }

        __device__ __host__ inline mat4 transpose() const {
            return {{x.x, y.x, z.x, w.x},
                    {x.y, y.y, z.y, w.y},
                    {x.z, y.z, z.z, w.z},
                    {x.w, y.w, z.w, w.w}};
        }


        __device__ __host__ inline float determinant() const {
            return mat4_determinant(x.x, x.y, x.z, x.w,
                                    y.x, y.y, y.z, y.w,
                                    z.x, z.y, z.z, z.w,
                                    w.x, w.y, w.z, w.w);
        }

        __device__ __host__ inline mat4 inverse() const {
            return transpose() / determinant();
        }

        __device__ __host__ inline mat4 adjoint() const {
            return mat4{
                    // First row of cofactors
                    vec4{
                            mat3_determinant(y.y, y.z, y.w, z.y, z.z, z.w, w.y, w.z, w.w),
                            -mat3_determinant(y.x, y.z, y.w, z.x, z.z, z.w, w.x, w.z, w.w),
                            mat3_determinant(y.x, y.y, y.w, z.x, z.y, z.w, w.x, w.y, w.w),
                            -mat3_determinant(y.x, y.y, y.z, z.x, z.y, z.z, w.x, w.y, w.z)
                    },
                    // Second row of cofactors
                    vec4{
                            -mat3_determinant(x.y, x.z, x.w, z.y, z.z, z.w, w.y, w.z, w.w),
                            mat3_determinant(x.x, x.z, x.w, z.x, z.z, z.w, w.x, w.z, w.w),
                            -mat3_determinant(x.x, x.y, x.w, z.x, z.y, z.w, w.x, w.y, w.w),
                            mat3_determinant(x.x, x.y, x.z, z.x, z.y, z.z, w.x, w.y, w.z)
                    },
                    // Third row of cofactors
                    vec4{
                            mat3_determinant(x.y, x.z, x.w, y.y, y.z, y.w, w.y, w.z, w.w),
                            -mat3_determinant(x.x, x.z, x.w, y.x, y.z, y.w, w.x, w.z, w.w),
                            mat3_determinant(x.x, x.y, x.w, y.x, y.y, y.w, w.x, w.y, w.w),
                            -mat3_determinant(x.x, x.y, x.z, y.x, y.y, y.z, w.x, w.y, w.z)
                    },
                    // Fourth row of cofactors
                    vec4{
                            -mat3_determinant(x.y, x.z, x.w, y.y, y.z, y.w, z.y, z.z, z.w),
                            mat3_determinant(x.x, x.z, x.w, y.x, y.z, y.w, z.x, z.z, z.w),
                            -mat3_determinant(x.x, x.y, x.w, y.x, y.y, y.w, z.x, z.y, z.w),
                            mat3_determinant(x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z)
                    }
            }.transpose();
        }

        __device__ __host__ inline mat4 cofactor() const {
            return adjoint();
        }
    };

    __device__ __host__ inline mat4 vec4::wedge() const {
        return mat4{
                {0,  -z, y,  -w},
                {z,  0,  -x, w},
                {-y, x,  0,  -w},
                {w,  -w, w,  0}
        };
    }

    __device__ __host__ inline mat4 vec4::outer(const vec4 &b) const {
        return mat4{
                {x * b.x, x * b.y, x * b.z, x * b.w},
                {y * b.x, y * b.y, y * b.z, y * b.w},
                {z * b.x, z * b.y, z * b.z, z * b.w},
                {w * b.x, w * b.y, w * b.z, w * b.w}
        };
    }
}

#endif //ENGINE24_CUDA__MAT_VEC_CUH
