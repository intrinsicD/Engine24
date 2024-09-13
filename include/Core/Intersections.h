//
// Created by alex on 18.07.24.
//

#ifndef ENGINE24_INTERSECTIONS_H
#define ENGINE24_INTERSECTIONS_H

#include "Frustum.h"
#include "AABBStruct.h"
#include "Line.h"
#include "OBB.h"
#include "Sphere.h"
#include "Triangle.h"

namespace Bcg{
    //------------------------------------------------------------------------------------------------------------------
    // AABB
    //------------------------------------------------------------------------------------------------------------------

    bool Intersect(const AABB &aabb, const Vector<float, 3> &point);

    bool Intersect(const AABB &aabb, const AABB &other);

    bool Intersect(const AABB &aabb, const Line &line);

    bool Intersect(const AABB &aabb, const OBB &obb);

    bool Intersect(const AABB &aabb, const Plane &plane);

    bool Intersect(const AABB &aabb, const Sphere &sphere);

    bool Intersect(const AABB &aabb, const Triangle &triangle);

    //------------------------------------------------------------------------------------------------------------------
    // Frustum
    //------------------------------------------------------------------------------------------------------------------
    bool Intersect(const Frustum &frustum, const Vector<float, 3> &point);

    bool Intersect(const Frustum &frustum, const AABB &aabb);

    bool Intersect(const Frustum &frustum, const Frustum &other);

    bool Intersect(const Frustum &frustum, const Line &line);

    bool Intersect(const Frustum &frustum, const OBB &obb);

    bool Intersect(const Frustum &frustum, const Plane &plane);

    bool Intersect(const Frustum &frustum, const Sphere &sphere);

    bool Intersect(const Frustum &frustum, const Triangle &triangle);
}

#endif //ENGINE24_INTERSECTIONS_H
