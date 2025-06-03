//
// Created by alex on 6/3/25.
//

#ifndef SPHEREUTILS_H
#define SPHEREUTILS_H

#include "Sphere.h"

namespace Bcg {
    template <typename T>
    Vector<T, 3> ClosestPoint(const Sphere<T> &sphere, const Vector<T, 3> &point) {
        return ClosestPointTraits<Sphere<T>, Vector<T, 3>>::closest_point(sphere, point);
    }




}
#endif //SPHEREUTILS_H
