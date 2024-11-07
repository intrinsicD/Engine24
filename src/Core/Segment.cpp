//
// Created by alex on 07.11.24.
//

#include "Segment.h"

namespace Bcg {
    Vector<float, 3> ClosestPoint(const Segment &segment, const Vector<float, 3> &point) {
        return Segment::closest_point(segment.start, segment.end, point);
    }

    float Distance(const Segment &segment, const Vector<float, 3> &point) {
        return Segment::distance(segment.start, segment.end, point);
    }

    float UnsignedDistance(const Segment &segment, const Vector<float, 3> &point) {
        return std::abs(Distance(segment, point));
    }
}