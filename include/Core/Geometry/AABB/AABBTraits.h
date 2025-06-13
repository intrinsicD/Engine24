//
// Created by alex on 13.06.25.
//

#ifndef ENGINE24_AABBTRAITS_H
#define ENGINE24_AABBTRAITS_H

#include "MatVec.h"

namespace Bcg {
    struct PointType;
    struct AABB;
    struct Sphere;
    struct Ray;
    struct Line;
    struct Plane;
    struct Segment;
    struct Triangle;

    struct AABBTraits {
        static PointType closest_point(const AABB &aabb, const PointType &point);

        static float distance(const AABB &aabb, const PointType &point){
            auto sqr_distance = squared_distance(aabb, point);
            return sqrt(sqr_distance);
        }

        static float squared_distance(const AABB &aabb, const PointType &point);

        static bool contains(const AABB &aabb, const PointType &point);

        static bool contains(const AABB &aabb, const AABB &other);

        static bool contains(const AABB &aabb, const Sphere &sphere);

        static bool contains(const AABB &aabb, const Segment &segment);

        static bool contains(const AABB &aabb, const Triangle &triangle);

        static bool intersects(const AABB &aabb, const AABB &other);

        static bool intersects(const AABB &aabb, const Sphere &sphere);

        static bool intersects(const AABB &aabb, const Ray &ray);

        static bool intersects(const AABB &aabb, const Line &line);

        static bool intersects(const AABB &aabb, const Plane &plane);

        static bool intersects(const AABB &aabb, const Segment &segment);

        static bool intersects(const AABB &aabb, const Triangle &triangle);
    };
}

#endif //ENGINE24_AABBTRAITS_H
