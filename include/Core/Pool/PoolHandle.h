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
        PoolHandle() : pool(nullptr), idx(std::numeric_limits<size_t>::max()) {}

        // Copy constructor
        PoolHandle(const PoolHandle &other) : pool(other.pool), idx(other.idx) {
            if (other.pool && idx < pool->properties.size()) {
                ++pool->properties.template get<size_t>(pool->ref_count_name)[idx];
            }
        }

        // Move constructor
        PoolHandle(PoolHandle &&other) noexcept : pool(other.pool), idx(other.idx) {
            other.pool = nullptr;
            other.idx = std::numeric_limits<size_t>::max();
        }

        PoolHandle& operator=(const PoolHandle& other) {
            if (this != &other) {
                // Decrement current reference
                if (pool && idx < pool->properties.size()) {
                    size_t &ref = pool->properties.template get<size_t>(pool->ref_count_name)[idx];
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
                if (pool && idx < pool->properties.size()) {
                    ++pool->properties.template get<size_t>(pool->ref_count_name)[idx];
                }
            }
            return *this;
        }

        PoolHandle& operator=(PoolHandle&& other) noexcept {
            if (this != &other) {
                // Decrement current reference
                if (pool && idx < pool->properties.size()) {
                    size_t &ref = pool->properties.template get<size_t>(pool->ref_count_name)[idx];
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
                other.pool = nullptr;
                other.idx = std::numeric_limits<size_t>::max();
            }
            return *this;
        }

        ~PoolHandle() {
            if (pool && idx < pool->properties.size()) {
                size_t &ref_count = pool->properties.template get<size_t>(pool->ref_count_name)[idx];
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
            return pool->properties.template get<size_t>(pool->ref_count_name)[idx];
        }

        operator T &() {
            return pool->properties.template get<T>(pool->objects_name)[idx];
        }

        operator const T &() const {
            return pool->properties.template get<T>(pool->objects_name)[idx];
        }

        T &operator*() {
            return pool->properties.template get<T>(pool->objects_name)[idx];
        }

        const T &operator*() const {
            return pool->properties.template get<T>(pool->objects_name)[idx];
        }

        T *operator->() {
            return &pool->properties.template get<T>(pool->objects_name)[idx];
        }

        const T *operator->() const {
            return &pool->properties.template get<T>(pool->objects_name)[idx];
        }

        bool operator==(const PoolHandle &other) const {
            return pool == other.pool && idx == other.idx;
        }

        bool operator!=(const PoolHandle &other) const {
            return !(*this == other);
        }

    protected:
        friend class Pool<T>;

        PoolHandle(Pool<T> *pool, size_t idx) : pool(pool), idx(idx) {
            if (pool && idx < pool->properties.size()) {
                ++pool->properties.template get_or_add<size_t>(pool->ref_count_name, 0)[idx];
            }
        }

        Pool<T> *pool;
        size_t idx;
    };
}

#endif //POOLHANDLE_H
