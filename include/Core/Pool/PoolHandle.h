//
// Created by alex on 11/17/24.
//

#ifndef POOLHANDLE_H
#define POOLHANDLE_H

#include <limits>

namespace Bcg {
    template<typename T>
    class Pool;

    template<typename T>
    struct PoolHandle {
        PoolHandle() : pool(nullptr), idx(std::numeric_limits<size_t>::max()), generation(0) {}

        // Copy constructor
        PoolHandle(const PoolHandle &other) : pool(other.pool), idx(other.idx), generation(other.generation) {
            if (other.pool && idx < pool->properties.size()) {
                ++pool->ref_count[idx];
            }
        }

        // Move constructor
        PoolHandle(PoolHandle &&other) noexcept : pool(other.pool), idx(other.idx), generation(other.generation) {
            other.pool = nullptr;
            other.idx = std::numeric_limits<size_t>::max();
            other.generation = 0;
        }

        size_t get_index() const {
            return idx;
        }

        size_t get_generation() const{
            return generation;
        }

        PoolHandle& operator=(const PoolHandle& other) {
            if (this != &other) {
                // Decrement current reference
                if (pool && idx < pool->properties.size()) {
                    size_t &ref = pool->ref_count[idx];
                    if (ref > 0) {
                        --ref;
                        if (ref == 0) {
                            pool->free_list.push(idx);
                        }
                    }
                }

                // Copy new reference
                pool = other.pool;
                idx = other.idx;
                generation = other.generation;
                if (pool && idx < pool->properties.size()) {
                    ++pool->ref_count[idx];
                }
            }
            return *this;
        }

        PoolHandle& operator=(PoolHandle&& other) noexcept {
            if (this != &other) {
                // Decrement current reference
                if (pool && idx < pool->properties.size()) {
                    size_t &ref = pool->ref_count[idx];
                    if (ref > 0) {
                        --ref;
                        if (ref == 0) {
                            pool->free_list.push(idx);
                        }
                    }
                }

                // Move new reference
                pool = other.pool;
                idx = other.idx;
                generation = other.generation;

                // Reset the moved-from handle
                other.pool = nullptr;
                other.idx = std::numeric_limits<size_t>::max();
                other.generation = 0;
            }
            return *this;
        }

        ~PoolHandle() {
            if (pool && idx < pool->properties.size()) {
                size_t &ref_count = pool->ref_count[idx];
                if (ref_count > 0) {
                    --ref_count;
                    if (ref_count == 0) {
                        pool->free_list.push(idx);
                    }
                }
            }
        }

        bool is_valid() const {
            return pool && idx < pool->properties.size() && get_ref_count() > 0;
        }

        size_t get_ref_count() const {
            return pool->ref_count[idx];
        }

        operator T &() {
            return pool->objects[idx];
        }

        operator const T &() const {
            return pool->objects[idx];
        }

        T &operator*() {
            return pool->objects[idx];
        }

        const T &operator*() const {
            return pool->objects[idx];
        }

        T *operator->() {
            return &pool->objects[idx];
        }

        const T *operator->() const {
            return &pool->objects[idx];
        }

        T *ptr() {
            return &pool->objects[idx];
        }

        const T *ptr() const {
            return &pool->objects[idx];
        }


        bool operator==(const PoolHandle &other) const {
            return pool == other.pool && idx == other.idx;
        }

        bool operator!=(const PoolHandle &other) const {
            return !(*this == other);
        }

    protected:
        friend class Pool<T>;

        PoolHandle(Pool<T> *pool, size_t idx, size_t generation) : pool(pool), idx(idx), generation(generation) {
            if (pool && idx < pool->properties.size()) {
                ++pool->ref_count[idx];
            }
        }

        Pool<T>* pool = nullptr;
        size_t idx = std::numeric_limits<size_t>::max();
        size_t generation = 0;
    };
}

#endif //POOLHANDLE_H
