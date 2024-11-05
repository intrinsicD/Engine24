//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_PROPERTYEIGENMAP_H
#define ENGINE24_PROPERTYEIGENMAP_H

#include "Properties.h"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include <type_traits>

namespace Bcg {
    template<typename T>
    auto Map(std::vector<T> &data, int rows, int cols) {
        return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >(data.data(), rows, cols);
    }

    template<typename T>
    auto MapConst(const std::vector<T> &data, int rows, int cols) {
        return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(data.data(), rows, cols);
    }

    template<typename T>
    inline Eigen::Map<Eigen::Vector<T, Eigen::Dynamic> > Map(std::vector<T> &data) {
        return Eigen::Map<Eigen::Vector<T, Eigen::Dynamic> >(data.data(), data.size());
    }

    template<typename T>
    inline Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> MapConst(const std::vector<T> &data) {
        return Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(data.data(), data.size());
    }

    template<typename T, int N>
    inline Eigen::Map<Eigen::Matrix<T, N, Eigen::Dynamic> > Map(std::vector<Eigen::Vector<T, N> > &data) {
        return Eigen::Map<Eigen::Matrix<T, N, Eigen::Dynamic> >(data.data(), N, data.size());
    }

    template<typename T, int N>
    inline Eigen::Map<const Eigen::Matrix<T, N, Eigen::Dynamic>>
    MapConst(const std::vector<Eigen::Vector<T, N> > &data) {
        return Eigen::Map<const Eigen::Matrix<T, N, Eigen::Dynamic>>(data.data(), N, data.size());
    }

    // Mapping a std::vector of glm::vec to an Eigen::Matrix
    template<typename T, int N, glm::qualifier Q = glm::defaultp>
    inline Eigen::Map<Eigen::Matrix<T, N, Eigen::Dynamic> > Map(std::vector<glm::vec<N, T, Q> > &data) {
        static_assert(sizeof(glm::vec<N, T, Q>) == N * sizeof(T),
                      "glm::vec<N, T, Q> has padding, cannot map directly.");
        return Eigen::Map<Eigen::Matrix<T, N, Eigen::Dynamic> >(reinterpret_cast<T *>(data.data()), N, data.size());
    }

    template<typename T, int N, glm::qualifier Q = glm::defaultp>
    inline Eigen::Map<const Eigen::Matrix<T, N, Eigen::Dynamic>> MapConst(const std::vector<glm::vec<N, T, Q> > &data) {
        static_assert(sizeof(glm::vec<N, T, Q>) == N * sizeof(T),
                      "glm::vec<N, T, Q> has padding, cannot map directly.");
        return Eigen::Map<const Eigen::Matrix<T, N, Eigen::Dynamic>>(reinterpret_cast<const T *>(data.data()), N,
                                                                     data.size());
    }
}

#endif //ENGINE24_PROPERTYEIGENMAP_H
