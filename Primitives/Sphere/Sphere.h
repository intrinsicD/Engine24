//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_SPHERE_H
#define ENGINE24_SPHERE_H

#include "MatVec.h"

namespace Bcg {
    template<typename T>
    struct SphereBase {
        Vector<T, 3> center;
        T radius;
    };

    using Spheref = SphereBase<float>;
    using Sphere = Spheref;

    Vector<float, 3> ClosestPoint(const Sphere &sphere, const Vector<float, 3> &point);

    float Volume(const Sphere &sphere);

    float SurfaceArea(const Sphere &sphere);

    float Distance(const Sphere &sphere, const Vector<float, 3> &point);

    float UnsignedDistance(const Sphere &sphere, const Vector<float, 3> &point);
}
#endif //ENGINE24_SPHERE_H
