//
// Created by alex on 24.10.24.
//

#ifndef ENGINE24_VECTOR_H
#define ENGINE24_VECTOR_H

#include "Macros.h"

namespace Bcg::Math {
    template<typename T, int N>
    struct Vector {
        T m_data[N];

        CUDA_HOST_DEVICE Vector() : m_data{0} {

        }

        CUDA_HOST_DEVICE Vector(T t) : m_data{t} {

        }

        CUDA_HOST_DEVICE Vector<T, N> Zeros() {
            return Constant(0);
        }

        CUDA_HOST_DEVICE Vector<T, N> Ones() {
            return Constant(1);
        }

        CUDA_HOST_DEVICE Vector<T, N> Constant(T t) {
            return {t};
        }

        CUDA_HOST_DEVICE Vector<T, N> Unit(int i) {
            Vector<T, N> v = Zeros();
            if (i < N) {
                v[i] = 1;
            }
            return v;
        }

        CUDA_HOST_DEVICE T &operator[](int i) {
            return m_data[i];
        }

        CUDA_HOST_DEVICE const T &operator[](int i) const {
            return m_data[i];
        }

        CUDA_HOST_DEVICE T *data() {
            return m_data[0];
        }

        CUDA_HOST_DEVICE const T *data() const {
            return m_data[0];
        }

        CUDA_HOST_DEVICE Vector<T, N> operator-() const {
            Vector<T, N> v;
            for (int i = 0; i < N; ++i) {
                v[i] = -m_data[i];
            }
            return v;
        }

        CUDA_HOST_DEVICE Vector<T, N> operator+(const Vector<T, N> &v) const {
            Vector<T, N> r;
            for (int i = 0; i < N; ++i) {
                r[i] = m_data[i] + v[i];
            }
            return r;
        }

        CUDA_HOST_DEVICE Vector<T, N> operator-(const Vector<T, N> &v) const {
            Vector<T, N> r;
            for (int i = 0; i < N; ++i) {
                r[i] = m_data[i] - v[i];
            }
            return r;
        }

        CUDA_HOST_DEVICE Vector<T, N> prod(const Vector<T, N> &v) const {
            Vector<T, N> r;
            for (int i = 0; i < N; ++i) {
                r[i] = m_data[i] * v[i];
            }
            return r;
        }

        CUDA_HOST_DEVICE T sum() const {
            T s = m_data[0];
            for (int i = 1; i < N; ++i) {
                s += m_data[i];
            }
            return s;
        }

        CUDA_HOST_DEVICE T prod() const {
            T p = m_data[0];
            for (int i = 1; i < N; ++i) {
                p *= m_data[i];
            }
            return p;
        }

        CUDA_HOST_DEVICE Vector<T, N> operator*(T t) const {
            Vector<T, N> r;
            for (int i = 0; i < N; ++i) {
                r[i] = m_data[i] * t;
            }
            return r;
        }

        CUDA_HOST_DEVICE Vector<T, N> operator/(T t) const {
            static_assert(t > 0, "Division by zero");
            Vector<T, N> r;
            for (int i = 0; i < N; ++i) {
                r[i] = m_data[i] / t;
            }
            return r;
        }

        CUDA_HOST_DEVICE Vector<T, N> &operator+=(const Vector<T, N> &v) {
            for (int i = 0; i < N; ++i) {
                m_data[i] += v[i];
            }
            return *this;
        }

        CUDA_HOST_DEVICE Vector<T, N> &operator-=(const Vector<T, N> &v) {
            for (int i = 0; i < N; ++i) {
                m_data[i] -= v[i];
            }
            return *this;
        }

        CUDA_HOST_DEVICE Vector<T, N> &operator*=(T t) {
            for (int i = 0; i < N; ++i) {
                m_data[i] *= t;
            }
            return *this;
        }

        CUDA_HOST_DEVICE Vector<T, N> &operator/=(T t) {
            static_assert(t > 0, "Division by zero");
            for (int i = 0; i < N; ++i) {
                m_data[i] /= t;
            }
            return *this;
        }

        CUDA_HOST_DEVICE bool operator==(const Vector<T, N> &v) const {
            for (int i = 0; i < N; ++i) {
                if (m_data[i] != v[i]) {
                    return false;
                }
            }
            return true;
        }

        CUDA_HOST_DEVICE bool operator!=(const Vector<T, N> &v) const {
            return !(*this == v);
        }

        CUDA_HOST_DEVICE T dot(const Vector<T, N> &v) const {
            T r = 0;
            for (int i = 0; i < N; ++i) {
                r += m_data[i] * v[i];
            }
            return r;
        }

        CUDA_HOST_DEVICE T squared_norm() const {
            return dot(*this);
        }

        CUDA_HOST_DEVICE T norm() const {
            return sqrt(squared_norm());
        }

        CUDA_HOST_DEVICE Vector<T, N> normalize(T eps = 1e-6) const {
            T n = norm();
            if (n < eps) {
                return *this / eps;
            } else {
                return *this / n;
            }
        }

        CUDA_HOST_DEVICE Vector<T, N> &normalized(T eps = 1e-6) {
            T n = norm();
            if (n < eps) {
                return *this /= eps;
            } else {
                return *this /= n;
            }
        }

        CUDA_HOST_DEVICE Vector<T, N - 1> head() const {
            static_assert(N > 1, "Vector dimension must be greater than 1");
            Vector<T, N - 1> v;
            for (int i = 0; i < N - 1; ++i) {
                v[i] = m_data[i];
            }
            return v;
        }

        CUDA_HOST_DEVICE Vector<T, N - 1> tail() const {
            static_assert(N > 1, "Vector dimension must be greater than 1");
            Vector<T, N - 1> v;
            for (int i = 0; i < N - 1; ++i) {
                v[i] = m_data[i + 1];
            }
            return v;
        }

        CUDA_HOST_DEVICE Vector<T, N+1> homogeneous() const {
            Vector<T, N+1> v;
            for (int i = 0; i < N; ++i) {
                v[i] = m_data[i];
            }
            v[N] = 1;
            return v;
        }
    };

    template<typename T, int N>
    CUDA_HOST_DEVICE Vector<T, N> operator*(T t, const Vector<T, N> &v) {
        return v * t;
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE Vector<T, N> operator/(T t, const Vector<T, N> &v) {
        Vector<T, N> r;
        for (int i = 0; i < N; ++i) {
            r[i] = t / v[i];
        }
        return r;
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE Vector<T, N> operator+(T t, const Vector<T, N> &v) {
        return v + t;
    }

    template<typename T, int N>
    CUDA_HOST_DEVICE Vector<T, N> operator-(T t, const Vector<T, N> &v) {
        return -v + t;
    }

    template<typename T>
    CUDA_HOST_DEVICE Vector<T, 3> cross(const Vector<T, 3> &v0, const Vector<T, 3> &v1) {
        return {v0[1] * v1[2] - v0[2] * v1[1],
                v0[2] * v1[0] - v0[0] * v1[2],
                v0[0] * v1[1] - v0[1] * v1[0]};
    }

    template<typename T>
    CUDA_HOST_DEVICE Vector<T, 2> perp(const Vector<T, 2> &v) {
        return {-v[1], v[0]};
    }
}

#endif //ENGINE24_VECTOR_H
