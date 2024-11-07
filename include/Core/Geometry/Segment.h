//
// Created by alex on 07.11.24.
//

#ifndef ENGINE24_SEGMENT_H
#define ENGINE24_SEGMENT_H

#include "MatVec.h"

namespace Bcg{
    template<typename T>
    struct SegmentBase {
        Vector<T, 3> start;
        Vector<T, 3> end;

        inline static Vector<T, 3> diff(const Vector<T, 3> &start, const Vector<T, 3> &end) {
            return end - start;
        }

        inline static T length(const Vector<T, 3> &start, const Vector<T, 3> &end) {
            return glm::length(diff(start, end));
        }

        inline static Vector<T, 3> direction(const Vector<T, 3> &start, const Vector<T, 3> &end) {
            return diff(start, end).normalized();
        }

        inline static Vector<T, 3> start_to_point(const Vector<T, 3> &start, const Vector<T, 3> &point) {
            return point - start;
        }

        inline static T parameter_dir(const Vector<T, 3> &start_to_point_, const Vector<T, 3> &dir) {
            return std::max(T(0), std::min(T(1), glm::dot(start_to_point_, dir) / glm::dot(dir, dir)));
        }

        inline static Vector<T, 3> closest_point(const Vector<T, 3> &start, const Vector<T, 3> &end, const Vector<T, 3> &point) {
            const Vector<T, 3> d = diff(start, end);
            const T t = parameter_dir(start_to_point(start, point), d);
            return start + t * d;
        }

        inline static T distance(const Vector<T, 3> &start, const Vector<T, 3> &end, const Vector<T, 3> &point) {
            return glm::length(closest_point(start, end, point) - point);
        }
    };

    using Segmentf = SegmentBase<float>;
    using Segment = Segmentf;

    //These functions forward to the inline static member functions of the SegmentBase struct
    Vector<float, 3> ClosestPoint(const Segment &segment, const Vector<float, 3> &point);

    float Distance(const Segment &segment, const Vector<float, 3> &point);

    float UnsignedDistance(const Segment &segment, const Vector<float, 3> &point);
}

#endif //ENGINE24_SEGMENT_H
