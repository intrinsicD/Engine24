//
// Created by alex on 04.06.25.
//

#ifndef ENGINE24_AABBSAMPLER_H
#define ENGINE24_AABBSAMPLER_H

#include "AABB.h"
#include <random>

namespace Bcg {
    template<typename T>
    std::vector<Vector<T, 3>> SampleSurfaceRandom(const AABB<T> &aabb, size_t num_samples) {
        std::vector<Vector<T, 3>> samples;
        samples.reserve(num_samples);
        for (size_t i = 0; i < num_samples; ++i) {
            T x = aabb.min[0] + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (aabb.max[0] - aabb.min[0])));
            T y = aabb.min[1] + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (aabb.max[1] - aabb.min[1])));
            T z = aabb.min[2] + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX / (aabb.max[2] - aabb.min[2])));
            samples.emplace_back(x, y, z);
        }
        return samples;
    }

    template<typename T>
    std::vector<Vector<T, 3>> SampleSurfaceUniform(const AABB<T> &aabb, size_t num_samples_per_axis) {
        std::vector<Vector<T, 3>> samples;
        samples.reserve(num_samples_per_axis * num_samples_per_axis * num_samples_per_axis);

        T step_x = (aabb.max[0] - aabb.min[0]) / (num_samples_per_axis - 1);
        T step_y = (aabb.max[1] - aabb.min[1]) / (num_samples_per_axis - 1);
        T step_z = (aabb.max[2] - aabb.min[2]) / (num_samples_per_axis - 1);

        for (size_t i = 0; i < num_samples_per_axis; ++i) {
            for (size_t j = 0; j < num_samples_per_axis; ++j) {
                for (size_t k = 0; k < num_samples_per_axis; ++k) {
                    T x = aabb.min[0] + i * step_x;
                    T y = aabb.min[1] + j * step_y;
                    T z = aabb.min[2] + k * step_z;
                    samples.emplace_back(x, y, z);
                }
            }
        }
        return samples;
    }

    template<typename T>
    std::vector<Vector<T, 3>> SampleVolumeRandom(const AABB<T> &aabb, size_t num_samples) {
        // Container for output samples
        std::vector<Vector<T, 3>> samples;
        samples.reserve(num_samples);

        // 1) Use std::random_device + mt19937 for good‐quality randomness
        std::random_device rd;
        std::mt19937 gen(rd());

        // 2) Set up independent uniform distributions for each axis:
        //    x ∼ Uniform[x_min, x_max]
        //    y ∼ Uniform[y_min, y_max]
        //    z ∼ Uniform[z_min, z_max]
        //
        //    Sampling each coordinate independently in its range yields a
        //    uniform distribution over the AABB’s volume. :contentReference[oaicite:0]{index=0}
        std::uniform_real_distribution<T> dist_x(aabb.min.x, aabb.max.x);
        std::uniform_real_distribution<T> dist_y(aabb.min.y, aabb.max.y);
        std::uniform_real_distribution<T> dist_z(aabb.min.z, aabb.max.z);

        // 3) Generate num_samples points by drawing each coordinate independently
        for (size_t i = 0; i < num_samples; ++i) {
            T x = dist_x(gen);
            T y = dist_y(gen);
            T z = dist_z(gen);
            samples.emplace_back(x, y, z);
        }

        return samples;
    }

    template<typename T>
    std::vector<Vector<T, 3>> SampleVolumeUniform(const AABB<T> &aabb, size_t num_samples_per_axis) {
        std::vector<Vector<T, 3>> samples;

        // If N = 0, return empty vector
        if (num_samples_per_axis == 0) {
            return samples;
        }

        // Total number of samples = N^3
        samples.reserve(num_samples_per_axis * num_samples_per_axis * num_samples_per_axis);

        // Compute the length of each axis: (x_max − x_min), (y_max − y_min), (z_max − z_min)
        T length_x = aabb.max.x - aabb.min.x;
        T length_y = aabb.max.y - aabb.min.y;
        T length_z = aabb.max.z - aabb.min.z;

        // If N = 1, place a single sample at the center of the box
        // (fallback to midpoint to avoid division by zero below)
        if (num_samples_per_axis == 1) {
            T mid_x = aabb.min.x + length_x * T(0.5);
            T mid_y = aabb.min.y + length_y * T(0.5);
            T mid_z = aabb.min.z + length_z * T(0.5);
            samples.emplace_back(mid_x, mid_y, mid_z);
            return samples;
        }

        // Compute spacing along each axis:
        // dx = (x_max − x_min) / (N − 1), etc.
        // This places the first point at x_min and the last at x_max, evenly spaced.
        T dx = length_x / T(num_samples_per_axis - 1);
        T dy = length_y / T(num_samples_per_axis - 1);
        T dz = length_z / T(num_samples_per_axis - 1);

        // Triple loop over i, j, k to place points on the N×N×N grid
        for (size_t i = 0; i < num_samples_per_axis; ++i) {
            T x = aabb.min.x + dx * T(i);
            for (size_t j = 0; j < num_samples_per_axis; ++j) {
                T y = aabb.min.y + dy * T(j);
                for (size_t k = 0; k < num_samples_per_axis; ++k) {
                    T z = aabb.min.z + dz * T(k);
                    samples.emplace_back(x, y, z);
                }
            }
        }

        return samples;
    }

}

#endif //ENGINE24_AABBSAMPLER_H
