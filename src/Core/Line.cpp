//
// Created by alex on 18.07.24.
//

#include "Line.h"

namespace Bcg {
    Vector<float, 3> ClosestPoint(const Line &line, const Vector<float, 3> &point) {
        return Line::closest_point(line.start, line.vec_to_end(), point, point - line.start);
    }

    float Distance(const Line &line, const Vector<float, 3> &point, const Vector<float, 3> start_to_point) {
        return Line::distance(line.start, line.vec_to_end(), point, start_to_point);
    }

    float Distance(const Line &line, const Vector<float, 3> &point) {
        return Distance(line, point, point - line.start);
    }

    float UnsignedDistance(const Line &line, const Vector<float, 3> &point) {
        return std::abs(Distance(line, point));
    }
}