//
// Created by alex on 12.07.24.
//

#ifndef ENGINE24_SPHERE_H
#define ENGINE24_SPHERE_H

#include "MatVec.h"
#include "StringTraits.h"
#include "VecTraits.h"
#include "Macros.h"
#include "Logger.h"
#include "GeometricTraits.h"

namespace Bcg {
    template<typename T>
    struct Sphere {
        Vector<T, 3> center = Vector<T, 3>(0, 0, 0);
        T radius = T(0);

        CUDA_HOST_DEVICE Sphere() = default;

        CUDA_HOST_DEVICE Sphere(const Vector<T, 3> &center, T radius) : center(center), radius(std::abs(radius)) {

        }

        CUDA_HOST_DEVICE Sphere(const Vector<T, 3> &point) : center(point), radius(T(0)) {

        }

        CUDA_HOST_DEVICE bool is_valid() const {
            return radius >= T(0);
        }

        CUDA_HOST_DEVICE void clear() {
            center = Vector<T, 3>(0, 0, 0);
            radius = T(0);
        }

        CUDA_HOST_DEVICE T volume() const {
            return M_PI * radius * radius * radius;
        }

        CUDA_HOST_DEVICE T surface_area() const {
            return 4 * M_PI * radius * radius;
        }
    };

    template<typename T>
    struct StringTraits<Sphere<T> > {
        static std::string ToString(const Sphere<T> &sphere) {
            std::stringstream ss;
            ss << MapConst(sphere.center).transpose() << ", " << sphere.radius;
            return ss.str();
        }
    };

     //------------------------------------------------------------------------------------------------------------------

    template<typename T>
    struct ClosestPointTraits<Sphere<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static Vector<T, 3> closest_point(const Sphere<T> &sphere, const Vector<T, 3> &point) noexcept {
            return VecTraits<Vector<T, 3>>::normalize(sphere.center - point) * sphere.radius + sphere.center;
        }
    };

    template<typename T>
    struct SquaredDistanceTraits<Sphere<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static T squared_distance(const Sphere<T> &sphere, const Vector<T, 3> &point) noexcept {
            return VecTraits<Vector<T, 3> >::squared_distance(
                ClosestPointTraits<Sphere<T>, Vector<T, 3> >::closest_point(sphere, point), point);
        }
    };

    template<typename T>
    struct DistanceTraits<Sphere<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static T distance(const Sphere<T> &sphere, const Vector<T, 3> &point) noexcept {
            return sqrt(SquaredDistanceTraits<Sphere<T>, Vector<T, 3> >::squared_distance(sphere, point));
        }
    };

    template<typename T>
    struct GetterTraits<Sphere<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static Sphere<T> getter(const Vector<T, 3> &v) noexcept {
            return {v, T(0)};
        }
    };

    template<typename T>
    CUDA_HOST_DEVICE static bool isWithinBounds(const Sphere<T> &a, const Vector<T, 3> &b) noexcept {
        return VecTraits<Vector<T, 3> >::squared_distance(a.center, b) <= a.radius * a.radius;
    }

    template<typename T>
    struct ContainsTraits<Sphere<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool contains(const Sphere<T> &a, const Vector<T, 3> &b) noexcept {
            return isWithinBounds(a, b);
        }
    };

    template<typename T>
    struct IntersectsTraits<Sphere<T>, Vector<T, 3> > {
        CUDA_HOST_DEVICE static bool intersects(const Sphere<T> &a, const Vector<T, 3> &b) noexcept {
            return isWithinBounds(a, b);
        }
    };

    template<typename T>
    struct ContainsTraits<Sphere<T>, Sphere<T> > {
        CUDA_HOST_DEVICE static bool contains(const Sphere<T> &a, const Sphere<T> &b) noexcept {
            return isWithinBounds(a, b.center) &&
                   VecTraits<Vector<T, 3> >::squared_distance(a.center, b.center) <= (a.radius - b.radius) * (a.radius - b.radius);
        }
    };

    template<typename T>
    struct IntersectsTraits<Sphere<T>, Sphere<T> > {
        CUDA_HOST_DEVICE static bool intersects(const Sphere<T> &a, const Sphere<T> &b) noexcept {
            return isWithinBounds(a, b.center) ||
                   isWithinBounds(b, a.center) ||
                   VecTraits<Vector<T, 3> >::squared_distance(a.center, b.center) <= (a.radius + b.radius) * (a.radius + b.radius);
        }
    };

    template<typename T>
    struct IntersectionTraits<Sphere<T>, Sphere<T> > {
        CUDA_HOST_DEVICE static Sphere<T> intersection(const Sphere<T> &a, const Sphere<T> &b) noexcept {
            Log::Warn("Intersection of two spheres is not implemented. Returning the first sphere.");
        }
    };
}
#endif //ENGINE24_SPHERE_H
