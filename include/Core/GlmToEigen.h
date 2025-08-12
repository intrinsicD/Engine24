#ifndef GLM_TO_EIGEN_H
#define GLM_TO_EIGEN_H

#include "Eigen/Core"
#include "MatVec.h"
#include "Types.h"
#include "glm/gtc/type_ptr.hpp" // For glm::value_ptr

namespace Bcg {
    // Mapping a glm::vec to an Eigen::Vector
    template<typename T, int N, glm::qualifier Q>
    inline Eigen::Map<Eigen::Matrix<T, N, 1>> Map(glm::vec<N, T, Q> &data) {
        return Eigen::Map<Eigen::Matrix<T, N, 1>>(glm::value_ptr(data), N);
    }

    template<typename T, int N, glm::qualifier Q>
    inline Eigen::Map<const Eigen::Matrix<T, N, 1>> MapConst(const glm::vec<N, T, Q> &data) {
        return Eigen::Map<const Eigen::Matrix<T, N, 1>>(glm::value_ptr(data), N);
    }

    // Mapping a glm::mat to an Eigen::Matrix
    template<typename T, int C, int R, glm::qualifier Q>
    inline Eigen::Map<Eigen::Matrix<T, R, C>> Map(glm::mat<C, R, T, Q> &data) {
        return Eigen::Map<Eigen::Matrix<T, R, C>>(glm::value_ptr(data), R, C);
    }

    template<typename T, int C, int R, glm::qualifier Q>
    inline Eigen::Map<const Eigen::Matrix<T, R, C>> MapConst(const glm::mat<C, R, T, Q> &data) {
        return Eigen::Map<const Eigen::Matrix<T, R, C>>(glm::value_ptr(data), R, C);
    }

    template<typename T, int N, glm::qualifier Q>
    inline Eigen::Vector<T, N> ToEigen(const glm::vec<N, T, Q> &v) {
        Eigen::Vector<T, N> result;
        for(int i = 0; i < N; ++i){
            result[i] = v[i];
        }
        return result;
    }

    template<typename T, int N>
    inline glm::vec<N, float, glm::defaultp> FromEigen(const Eigen::Vector<T, N> &v) {
        glm::vec<N, float, glm::defaultp> result;
        for(int i = 0; i < v.size(); ++i){
            result[i] = v[i];
        }
        return result;
    }

    template<typename T, int N, glm::qualifier Q>
    inline T *DataPtr(glm::vec<N, T, Q> &data) {
        return glm::value_ptr(data);
    }

    template<typename T, int N>
    inline T *DataPtr(Eigen::Vector<T, N> &data) {
        return data.data();
    }

    template<typename T, int C, int R, glm::qualifier Q>
    inline T *DataPtr(glm::mat<C, R, T, Q> &data) {
        return glm::value_ptr(data);
    }

    template<typename T, int M, int N>
    inline T *DataPtr(Eigen::Matrix<T, M, N> &data) {
        return data.data();
    }

    template<typename T, int N, glm::qualifier Q>
    inline const T *DataPtr(const glm::vec<N, T, Q> &data) {
        return glm::value_ptr(data);
    }

    template<typename T, int N>
    inline const T *DataPtr(const Eigen::Vector<T, N> &data) {
        return data.data();
    }

    template<typename T, int C, int R, glm::qualifier Q>
    inline const T *DataPtr(const glm::mat<C, R, T, Q> &data) {
        return glm::value_ptr(data);
    }

    template<typename T, int M, int N>
    inline const T *DataPtr(const Eigen::Matrix<T, M, N> &data) {
        return data.data();
    }

    inline Eigen::Matrix3f ToEigen(const Matrix<float, 3, 3> &m) {
        // Assuming your Matrix is column-major like OpenGL/GLM
        return Eigen::Map<const Eigen::Matrix3f>(DataPtr(m));
    }

    template<typename T>
    inline Matrix<float, 3, 3> FromEigen(const Eigen::Matrix<T, 3, 3> &m) {
        Matrix<float, 3, 3> result;
        // Assuming your Matrix is column-major
        Eigen::Map<Eigen::Matrix<float, 3, 3>>(DataPtr(result)) = m.template cast<float>();
        return result;
    }
}

#endif //GLM_TO_EIGEN_H
