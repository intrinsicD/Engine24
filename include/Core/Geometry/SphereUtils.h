//
// Created by alex on 21.05.25.
//

#ifndef ENGINE24_SPHEREUTILS_H
#define ENGINE24_SPHEREUTILS_H

#include "Sphere.h"
#include "StringTraits.h"

namespace Bcg::SphereUtils {
    template<typename T, int N>
    Eigen::Vector<T, N> ClosestPoint(const Sphere<T, N> &sphere, const Eigen::Vector<T, N> &point) {
        return sphere.center + (point - sphere.center).normalized() * sphere.radius;
    }

    template<typename T, int N>
    T Distance(const Sphere<T, N> &sphere, const Eigen::Vector<T, N> &point) {
        return (point - sphere.center).norm() - sphere.radius;
    }
}

namespace Bcg {
    /**
     * @brief String representation of the AABB.
     * @param aabb The Axis-Aligned Bounding Box.
     * @return A string representation of the AABB.
     */
    template<typename S, int N>
    struct StringTraits<Sphere<S, N>> {
    static std::string ToString(const Sphere<S, N> &sphere) {
        std::stringstream ss;
        ss << "Sphere(center=" << sphere.center.transpose() << ", radius=" << sphere.radius << ")";
        return ss.str();
    }
};
}

#endif //ENGINE24_SPHEREUTILS_H
