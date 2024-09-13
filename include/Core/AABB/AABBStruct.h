//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_AABBSTRUCT_H
#define ENGINE24_AABBSTRUCT_H

#include "MatVec.h"
#include <limits>

namespace Bcg {
    template<typename T, int N>
    struct AABBStruct {
        Vector<T, N> min;
        Vector<T, N> max;

        AABBStruct() : min(Vector<T, N>::Constant(std::numeric_limits<T>::max())),
                       max(Vector<T, N>::Constant(std::numeric_limits<T>::lowest())) {}

        template<typename InputIterator>
        AABBStruct(InputIterator first, InputIterator last) : AABBStruct() {
            for (auto it = first; it != last; ++it) {
                grow(*it);
            }
        }

        explicit AABBStruct(const std::vector<Vector<T, N>> &points) : AABBStruct(points.begin(), points.end()) {}

        bool is_valid() const {
            return (min.array() <= max.array()).all();
        }

        void grow(const Vector<T, N> &point) {
            min = min.cwiseMin(point);
            max = max.cwiseMax(point);
        }

        Vector<T, N> diagonal() const {
            return max - min;
        }

        Vector<T, N> half_extent() const {
            return diagonal() / 2;
        }

        Vector<T, N> center() const {
            return (min + max) / 2;
        }

        T volume() const {
            return diagonal().prod();
        }
    };

    using AABBf = AABBStruct<float, 3>;
    using AABB = AABBf;
}

#endif //ENGINE24_AABBSTRUCT_H
