//
// Created by alex on 13.06.25.
//

#ifndef ENGINE24_DUMBHANDLE_H
#define ENGINE24_DUMBHANDLE_H

#include <functional>

namespace Bcg{
    struct DumbHandle{
        size_t index = -1; // Unique identifier for the handle
        size_t generation = 0; // Generation count to handle invalidation

        // Default constructor for an invalid handle
        DumbHandle() = default;

        // Explicit constructor
        DumbHandle(size_t idx, size_t gen) : index(idx), generation(gen) {}

        // For using in maps and comparisons
        bool operator==(const DumbHandle& other) const {
            return index == other.index && generation == other.generation;
        }
    };
}

namespace std {
    template <>
    struct hash<Bcg::DumbHandle> {
        size_t operator()(const Bcg::DumbHandle& h) const {
            // A simple hash combining both fields
            return hash<size_t>()(h.index) ^ (hash<size_t>()(h.generation) << 1);
        }
    };
}

#endif //ENGINE24_DUMBHANDLE_H
