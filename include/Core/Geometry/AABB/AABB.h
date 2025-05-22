//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_AABB_H
#define ENGINE24_AABB_H

#include "MatVec.h"
#include "Macros.h"

namespace Bcg {
    template<typename T, int N>
    struct AABB {
        Eigen::Vector<T, N> min = Eigen::Vector<T, N>::Constant(std::numeric_limits<T>::max());
        Eigen::Vector<T, N> max = Eigen::Vector<T, N>::Constant(std::numeric_limits<T>::lowest());

        CUDA_HOST_DEVICE static AABB FromPoint(const Eigen::Vector<T, N> &point) {
            return {point, point};
        }

        template<typename Iterator>
        CUDA_HOST static AABB Build(const Iterator &begin, const Iterator &end) {
            AABB result;
            for (auto it = begin; it != end; ++it) {
                result.grow(*it);
            }
            return result;
        }

        CUDA_HOST_DEVICE void clear() {
            min = Eigen::Vector<T, N>::Constant(std::numeric_limits<T>::max());
            max = Eigen::Vector<T, N>::Constant(std::numeric_limits<T>::lowest());
        }

        CUDA_HOST_DEVICE void merge(const AABB &other) {
            min = min.cwiseMin(other.min);
            max = max.cwiseMax(other.max);
        }

        CUDA_HOST_DEVICE void grow(const Eigen::Vector<T, N> &point) {
            min = min.cwiseMin(point);
            max = max.cwiseMax(point);
        }

        CUDA_HOST_DEVICE Eigen::Vector<T, N> diagonal() const {
            return max - min;
        }

        CUDA_HOST_DEVICE Eigen::Vector<T, N> half_extent() const {
            return diagonal() * 0.5;
        }

        CUDA_HOST_DEVICE Eigen::Vector<T, N> center() const {
            return (min + max) * 0.5;
        }

        CUDA_HOST_DEVICE T volume() const {
            return diagonal().prod();
        }
    };


}

#endif //ENGINE24_AABB_H
