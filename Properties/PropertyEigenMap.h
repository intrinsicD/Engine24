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
    template<typename T, typename Enable = void>
    struct MapTraits;

    template<typename Scalar, int Rows>
    struct MapTraits<Eigen::Matrix<Scalar, Rows, 1>, typename std::enable_if<Rows != Eigen::Dynamic>::type> {
        using Type = Eigen::Map<Eigen::Matrix<Scalar, Rows, Eigen::Dynamic>>;
    };

    template<typename Scalar, int Cols>
    struct MapTraits<Eigen::Matrix<Scalar, 1, Cols>, typename std::enable_if<Cols != Eigen::Dynamic>::type> {
        using Type = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Cols>>;
    };

// Specialization for scalar types
    template<typename Scalar>
    struct MapTraits<Scalar, typename std::enable_if<std::is_arithmetic<Scalar>::value>::type> {
        using Type = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;
    };

    template<typename T>
    auto Map(std::vector<T> &p) {
        using MatrixType = typename MapTraits<T>::Type;
        if constexpr (std::is_arithmetic<T>::value) {
            return MatrixType(p, p.size(), 1);
        }else if constexpr (T::RowsAtCompileTime != 1){
            return MatrixType(&p[0][0], T::RowsAtCompileTime, p.size());
        }else if constexpr (T::ColsAtCompileTime != 1){
            return MatrixType(&p[0][0],  p.size(), T::ColsAtCompileTime);
        }
    }
}

#endif //ENGINE24_PROPERTYEIGENMAP_H
