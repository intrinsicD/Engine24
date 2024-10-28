//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_LINE_H
#define ENGINE24_LINE_H

#include "MatVec.h"

namespace Bcg {
    template<typename T>
    struct LineBase {
        Vector<T, 3> start;
        Vector<T, 3> end;

        Vector<T, 3> vec_to_end() const {
            return end - start;
        }

        Vector<T, 3> direction() const {
            return vec_to_end().normalized();
        }

        static T parameter(const Vector<T, 3> &diff, const Vector<T, 3> start_to_point) {
            return glm::dot(start_to_point, diff) / glm::dot(diff, diff);
        }

        static Vector<T, 3> closest_point(const Vector<T, 3> start, const Vector<T, 3> &diff,
                                          const Vector<T, 3> &point, const Vector<T, 3> start_to_point) {
            T t = std::max(T(0), std::min(T(1), parameter(diff, start_to_point)));
            return point - (start + t * diff);
        }

        static T distance(const Vector<T, 3> start, const Vector<T, 3> &vec_to_end,
                          const Vector<T, 3> &point, const Vector<T, 3> start_to_point) {
            return glm::length(closest_point(start, vec_to_end, point, start_to_point));
        }
    };

    using Linef = LineBase<float>;
    using Line = Linef;

    Vector<float, 3> ClosestPoint(const Line &line, const Vector<float, 3> &point);

    float Distance(const Line &line, const Vector<float, 3> &point, const Vector<float, 3> start_to_point);

    float Distance(const Line &line, const Vector<float, 3> &point);

    float UnsignedDistance(const Line &line, const Vector<float, 3> &point);
}

#endif //ENGINE24_LINE_H
