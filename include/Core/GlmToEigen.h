#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp" // For glm::value_ptr
#include "Eigen/Core"
#include "MatVec.h"

namespace Bcg {
    // Mapping a glm::vec to an Eigen::Vector
    template<typename T, int N, glm::qualifier Q = glm::defaultp>
    inline Eigen::Map<Eigen::Matrix<T, N, 1>> Map(glm::vec<N, T, Q> &data) {
        return Eigen::Map<Eigen::Matrix<T, N, 1>>(glm::value_ptr(data), N);
    }

    template<typename T, int N, glm::qualifier Q = glm::defaultp>
    inline Eigen::Map<const Eigen::Matrix<T, N, 1>> MapConst(const glm::vec<N, T, Q> &data) {
        return Eigen::Map<const Eigen::Matrix<T, N, 1>>(glm::value_ptr(data), N);
    }

    // Mapping a glm::mat to an Eigen::Matrix
    template<typename T, int C, int R, glm::qualifier Q = glm::defaultp>
    inline Eigen::Map<Eigen::Matrix<T, R, C>> Map(glm::mat<C, R, T, Q> &data) {
        return Eigen::Map<Eigen::Matrix<T, R, C>>(glm::value_ptr(data), R, C);
    }

    template<typename T, int C, int R, glm::qualifier Q = glm::defaultp>
    inline Eigen::Map<const Eigen::Matrix<T, R, C>> MapConst(const glm::mat<C, R, T, Q> &data) {
        return Eigen::Map<const Eigen::Matrix<T, R, C>>(glm::value_ptr(data), R, C);
    }
}
