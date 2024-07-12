//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_LINE_H
#define ENGINE24_LINE_H

#include "../MatVec.h"

namespace Bcg{
    template<typename T>
    struct Line {
        Vector<T, 3> start;
        Vector<T, 3> vec_to_end;

        static T parameter(const Vector<T, 3> &vec_to_end, const Vector<T, 3> start_to_point) {
            return start_to_point.dot(vec_to_end) / vec_to_end.dot(vec_to_end);
        }

        static Vector<T, 3> closest_point(const Vector<T, 3> start, const Vector<T, 3> &vec_to_end,
                                          const Vector<T, 3> &point, const Vector<T, 3> start_to_point) {
            T t = std::max(T(0), std::min(T(1), parameter(vec_to_end, start_to_point)));
            return point - (start + t * vec_to_end);
        }

        static T distance(const Vector<T, 3> start, const Vector<T, 3> &vec_to_end,
                          const Vector<T, 3> &point, const Vector<T, 3> start_to_point) {
            return closest_point(start, vec_to_end, point, start_to_point).norm();
        }
    };

    template<typename T>
    Vector<T, 3> closest_point(const Line<T> &line, const Vector<T, 3> &point) {
        return Line<T>::closest_point(line.start, line.vec_to_end, point, point - line.start);
    }

    template<typename T>
    T distance(const Line<T> &line, const Vector<T, 3> &point, const Vector<T, 3> start_to_point) {
        return Line<T>::distance(line.start, line.vec_to_end, point, start_to_point);
    }

    template<typename T>
    T distance(const Line<T> &line, const Vector<T, 3> &point) {
        return distance(line, point, point - line.start);
    }
}

#endif //ENGINE24_LINE_H
