//
// Created by alex on 6/3/25.
//

#ifndef SPHEREUTILS_H
#define SPHEREUTILS_H

#include "Sphere.h"
#include "SurfaceMesh.h"

namespace Bcg {
    template <typename T>
    Vector<T, 3> ClosestPoint(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return ClosestPointTraits<Sphere<T>, Vector<T, 3>>::closest_point(sphere, point);
    }

    template<typename T>
    T SquaredDistance(const Sphere<T> &sphere, const Vector<T, 3> &point){
        return SquaredDistanceTraits<Sphere<T>, Vector<T, 3>>::squared_distance(sphere, point);
    }

    template<typename T>
    T Distance(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return DistanceTraits<Sphere<T>, Vector<T, 3>>::distance(sphere, point);
    }

    template<typename T>
    Sphere<T> Build(const Vector<T, 3> &v) {
        return BuilderTraits<Sphere<T>, Vector<T, 3>>::build(v);
    }

    template<typename T>
    bool Contains(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return ContainsTraits<Sphere<T>, Vector<T, 3>>::contains(sphere, point);
    }

    template<typename T>
    bool Intersects(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return IntersectsTraits<Sphere<T>, Vector<T, 3>>::intersects(sphere, point);
    }

    template<typename T>
    bool Contains(const Sphere<T> &a, const Sphere<T> &b) {
        return ContainsTraits<Sphere<T>, Sphere<T>>::contains(a, b);
    }

    template<typename T>
    bool Intersects(const Sphere<T> &a, const Sphere<T> &b) {
        return IntersectsTraits<Sphere<T>, Sphere<T>>::intersects(a, b);
    }
}
#endif //SPHEREUTILS_H
