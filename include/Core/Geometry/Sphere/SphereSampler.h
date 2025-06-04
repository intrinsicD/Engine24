//
// Created by alex on 04.06.25.
//

#ifndef ENGINE24_SPHERESAMPLER_H
#define ENGINE24_SPHERESAMPLER_H

#include "Sphere.h"
#include <random>

namespace Bcg {
    template<typename T>
    std::vector<Vector<T, 3>> SampleSurfaceRandom(const Sphere<T> &sphere, size_t num_samples) {
        std::vector<Vector<T, 3>> points(num_samples);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-1.0, 1.0);

        // Generate random points inside the unit sphere and then scale them to the sphere's radius
        // Ensure the points are uniformly distributed over the sphere's surface
        for (size_t i = 0; i < num_samples; ++i) {
            Vector<T, 3> point;
            do {
                point = {dist(gen), dist(gen), dist(gen)};
            } while (point.squaredNorm() > 1.0); // Ensure the point is inside the unit sphere
            points[i] = point.normalized() * sphere.radius + sphere.center;
        }
        return points;
    }

    template<typename T>
    std::vector<Vector<T, 3>> SampleSurfaceUniform(const Sphere<T> &sphere, size_t num_samples) {
        std::vector<Vector<T, 3>> points(num_samples);
        T phi = M_PI * (3.0 - std::sqrt(5.0)); // Golden angle in radians
        T radius = sphere.radius;

        for (size_t i = 0; i < num_samples; ++i) {
            T y = 1 - (i / float(num_samples - 1)) * 2; // y goes from 1 to -1
            T radius_at_y = std::sqrt(1 - y * y); // Radius at this y level
            T theta = phi * i; // Golden angle increment

            points[i] = {
                sphere.center.x + radius_at_y * std::cos(theta) * radius,
                sphere.center.y + radius_at_y * std::sin(theta) * radius,
                sphere.center.z + y * radius
            };
        }
        return points;
    }

    enum FibonacciLattice {
        FLNAIVE,    // No offset; samples are placed directly from i = 0 to N−1.
        FLFIRST,    // Offset by 0.5 to avoid including exact poles, improving uniformity.
        FLSECOND,   // Offset by 1.5 and adjust total count by 2*offset for a different distribution.
        FLTHIRD,    // Offset by 3.5, skip first and last index to place poles explicitly.
        FLOFFSET    // Use a small epsilon (~0.36) to fine‐tune pole spacing.
    };

    namespace helper{
        template<typename T>
        Vector<T, 3> LatticePoint(size_t i, size_t num_samples, T golden_ratio, T TWOPI, T index_offset,
                                  T sample_count_offset, const Sphere<T> &sphere) {
            T x = (i + index_offset) / T(num_samples + sample_count_offset);
            T y = T(i) / golden_ratio;
            T phi = std::acos(1.0 - 2.0 * x);
            T theta = TWOPI * y;
            Vector<T, 3> point = {std::cos(theta) * std::sin(phi), std::sin(theta) * std::sin(phi),
                                  std::cos(phi)};
            return point * sphere.radius + sphere.center;
        }
    }


    //http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/
    //Thou et. al. - 2024 - https://arxiv.org/pdf/2410.12007v1
    //Alvaro Gonzalez - 2009 - https://arxiv.org/pdf/0912.4540
    template<typename T>
    std::vector<Vector<T, 3>> SampleSurfaceFibonacciLattice(const Sphere<T> &sphere, size_t num_samples,
                                                     FibonacciLattice type = FLTHIRD) {
        //http://extremelearning.com.au/evenly-distributing-points-on-a-sphere/
        std::vector<Vector<T, 3>> points(num_samples);

        T golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;
        T TWOPI = 2 * M_PI;
        T epsilon = 0.36;
        size_t start_index = 0;
        size_t end_index = num_samples;
        T offset = 0.0;
        T sample_count_offset = 0.0;
        switch (type) {
            default:
            case FLNAIVE: {
                offset = 0.0;
                sample_count_offset = 0.0;
                break;
            }
            case FLFIRST: {
                offset = 0.5;
                sample_count_offset = 0.0;
                break;
            }
            case FLSECOND: {
                offset = 1.5;
                sample_count_offset = 2 * offset;
                break;
            }
            case FLTHIRD: {
                offset = 3.5;
                sample_count_offset = 2 * offset;
                start_index = 1;
                end_index = num_samples - 1;
                break;
            }
            case FLOFFSET: {
                offset = epsilon;
                sample_count_offset = 2 * offset - 1;
                start_index = 1;
                end_index = num_samples - 1;
                break;
            }
        }

        if (type == FLTHIRD || type == FLOFFSET) {
            points[start_index] = Vector<T, 3>(0, 0, 1) * sphere.radius + sphere.center;
        }
        for (size_t i = start_index; i < end_index; ++i) {
            points[i] = helper::LatticePoint(i, num_samples, golden_ratio, TWOPI, offset, sample_count_offset, sphere);
        }
        if (type == FLTHIRD || type == FLOFFSET) {
            points[end_index] = Vector<T, 3>(0, 0, -1) * sphere.radius + sphere.center;
        }
        return points;
    }

    //Aaron R. Voelker, Jan Gosmann, Terrence C. Stewart. “Efficiently sampling vectors and coordinates from the n‐sphere and n‐ball,” Centre for Theoretical Neuroscience, University of Waterloo (2017).
    //“Uniformly at random within the n‐ball,” Wikipedia (accessed May 2025).
    template<typename T>
    std::vector<Vector<T, 3>> SampleVolumeRandom(const Sphere<T> &sphere, size_t num_samples) {
        std::vector<Vector<T, 3>> points(num_samples);

        // 1) random_device + mt19937 for high‐quality randomness
        std::random_device rd;
        std::mt19937 gen(rd());

        // 2) Generate uniform real in [-1,1] (to be scaled by sphere.radius)
        std::uniform_real_distribution<T> dist(-1.0, 1.0);

        // 3) Rejection sampling in the cube [−R,R]^3:
        //    - Draw (x,y,z) in [−1,1]^3, scale by R
        //    - Reject if x^2+y^2+z^2 > R^2
        //    This yields a uniform distribution over the sphere’s volume :contentReference[oaicite:0]{index=0}
        for (size_t i = 0; i < num_samples; ++i) {
            Vector<T, 3> point;
            do {
                // a) random sample in unit cube
                point = { dist(gen), dist(gen), dist(gen) };
                // b) scale to actual sphere radius
                point *= sphere.radius;
                // c) check if inside sphere: squaredNorm() <= R^2
            } while (point.squaredNorm() > (sphere.radius * sphere.radius));

            // 4) Translate to sphere center
            points[i] = point + sphere.center;
        }

        return points;
    }

    //Aaron R. Voelker, Jan Gosmann, Terrence C. Stewart. “Efficiently sampling vectors and coordinates from the n‐sphere and n‐ball,” Centre for Theoretical Neuroscience, University of Waterloo (2017).
    //“Uniformly at random within the n‐ball,” Wikipedia (accessed May 2025).
    //“Walk-on-spheres method,” Wikipedia (accessed May 2025).
    template<typename T>
    std::vector<Vector<T, 3>> SampleVolumeUniform(const Sphere<T> &sphere, size_t num_samples) {
        std::vector<Vector<T, 3>> points;
        points.reserve(num_samples);

        // 1) Use std::random_device + mt19937 for high‐quality randomness
        std::random_device rd;
        std::mt19937 gen(rd());

        // 2) We'll need three independent Uniform[0,1] draws (u, u', u''):
        std::uniform_real_distribution<T> dist01(0.0, 1.0);

        for (size_t i = 0; i < num_samples; ++i) {
            // a) Draw u ∼ Uniform[0,1] for radial coordinate:
            //    r = sphere.radius * ∛u ensures uniformity in volume (Jacobian ∝ r^2).
            //    :contentReference[oaicite:0]{index=0}
            T u  = dist01(gen);
            T r  = sphere.radius * std::cbrt(u);

            // b) Draw u' ∼ Uniform[0,1] to get cosθ ∈ [−1,1]:
            //    cosθ = 1 − 2u' ⇒ θ has pdf ∝ sinθ, giving uniform distribution on the sphere’s surface :contentReference[oaicite:1]{index=1}
            T u_prime   = dist01(gen);
            T cos_theta = 1.0 - T(2.0) * u_prime;
            T sin_theta = std::sqrt(std::max<T>(0, T(1.0) - cos_theta * cos_theta));  // guard against tiny negatives

            // c) Draw u'' ∼ Uniform[0,1] to get φ ∈ [0,2π):
            //    φ = 2π·u''  gives uniform azimuthal angle. :contentReference[oaicite:2]{index=2}
            T u_doubleprime = dist01(gen);
            T phi = T(2.0) * M_PI * u_doubleprime;

            // d) Convert (r, θ, φ) → Cartesian unit‐vector * r:
            //
            //    x = r · sinθ · cosφ
            //    y = r · sinθ · sinφ
            //    z = r · cosθ
            Vector<T, 3> dir = {
                    sin_theta * std::cos(phi),
                    sin_theta * std::sin(phi),
                    cos_theta
            };

            // e) Scale by r and translate by sphere.center:
            points.push_back(dir * r + sphere.center);
        }

        return points;
    }

}

#endif //ENGINE24_SPHERESAMPLER_H
