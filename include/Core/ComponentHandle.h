#pragma once

#include <cstddef>

namespace Bcg {
    template<typename T>
    struct ComponentHandle {
        ComponentHandle(T &p, size_t gen) : ptr(&p), generation(gen) {
        }

        ComponentHandle(T *p, size_t gen) : ptr(p), generation(gen) {
        }

        ComponentHandle(T &p) : ptr(&p), generation(0) {
        }

        ComponentHandle(T *p) : ptr(p), generation(0) {
        }

        ComponentHandle() : ptr(nullptr), generation(0) {
        }

        T *ptr = nullptr;
        size_t generation = 0;

        operator bool() const {
            return ptr != nullptr;
        }

        bool operator==(const ComponentHandle &other) const {
            return ptr == other.ptr && generation == other.generation;
        }

        bool operator!=(const ComponentHandle &other) const {
            return ptr != other.ptr || generation != other.generation;
        }

        T *operator->() {
            return ptr;
        }
    };
}
