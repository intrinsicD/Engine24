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
        return Eigen::Map<Eigen::Matrix<T, N, Eigen::Dynamic> >(data[0].data(), N, data.size());
    }

    template<typename T, int N>
    inline Eigen::Map<const Eigen::Matrix<T, N, Eigen::Dynamic>>
    MapConst(const std::vector<Eigen::Vector<T, N> > &data) {
        return Eigen::Map<const Eigen::Matrix<T, N, Eigen::Dynamic>>(data[0].data(), N, data.size());
    }
}

#endif //ENGINE24_PROPERTYEIGENMAP_H
