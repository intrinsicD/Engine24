#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp" // For glm::value_ptr
#include "Eigen/Core"

namespace Bcg {
    // Mapping a glm::vec to an Eigen::Vector
    template<typename T, int N, glm::qualifier Q = glm::defaultp>
    Eigen::Map<Eigen::Matrix<T, N, 1>> ToEigen(glm::vec<N, T, Q> &data) {
        return Eigen::Map<Eigen::Matrix<T, N, 1>>(glm::value_ptr(data), N);
    }

    template<typename T, int N, glm::qualifier Q = glm::defaultp>
    Eigen::Map<const Eigen::Matrix<T, N, 1>> ToEigen(const glm::vec<N, T, Q> &data) {
        return Eigen::Map<const Eigen::Matrix<T, N, 1>>(glm::value_ptr(data), N);
    }

    // Mapping a glm::mat to an Eigen::Matrix
    template<typename T, int N, int M, glm::qualifier Q = glm::defaultp>
    Eigen::Map<Eigen::Matrix<T, N, M>> ToEigen(glm::mat<N, M, T, Q> &data) {
        return Eigen::Map<Eigen::Matrix<T, N, M>>(glm::value_ptr(data), N, M);
    }

    template<typename T, int N, int M, glm::qualifier Q = glm::defaultp>
    Eigen::Map<const Eigen::Matrix<T, N, M>> ToEigen(const glm::mat<N, M, T, Q> &data) {
        return Eigen::Map<const Eigen::Matrix<T, N, M>>(glm::value_ptr(data), N, M);
    }

    // Mapping a std::vector of glm::vec to an Eigen::Matrix
    template<typename T, int N, glm::qualifier Q = glm::defaultp>
    Eigen::Map<Eigen::Matrix<T, N, Eigen::Dynamic>> ToEigen(std::vector<glm::vec<N, T, Q>> &data) {
        static_assert(sizeof(glm::vec<N, T, Q>) == N * sizeof(T), "glm::vec<N, T, Q> has padding, cannot map directly.");
        return Eigen::Map<Eigen::Matrix<T, N, Eigen::Dynamic>>(reinterpret_cast<T*>(data.data()), N, data.size());
    }

    template<typename T, int N, glm::qualifier Q = glm::defaultp>
    Eigen::Map<const Eigen::Matrix<T, N, Eigen::Dynamic>> ToEigen(const std::vector<glm::vec<N, T, Q>> &data) {
        static_assert(sizeof(glm::vec<N, T, Q>) == N * sizeof(T), "glm::vec<N, T, Q> has padding, cannot map directly.");
        return Eigen::Map<const Eigen::Matrix<T, N, Eigen::Dynamic>>(reinterpret_cast<const T*>(data.data()), N, data.size());
    }
}
