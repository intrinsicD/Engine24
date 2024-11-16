//
// Created by alex on 30.10.24.
//
#ifndef ENGINE24_POOL_H
#define ENGINE24_POOL_H

#include "Properties.h"
#include <queue>
#include <limits>

namespace Bcg {
    template<typename T>
    struct PoolHandle;

    template<typename T>
    class Pool {
    public:
        Pool() : ref_count(properties.get_or_add<size_t>(ref_count_name, 0)),
                 objects(properties.get_or_add<T>(objects_name)) {

        }

        virtual ~Pool() = default;

        PropertyContainer properties;
        Property<size_t> ref_count;
        Property<T> objects;

        PoolHandle<T> create() {
            size_t idx;
            if (free_list.empty()) {
                properties.push_back();
                idx = properties.size() - 1;
            } else {
                idx = free_list.front();
                free_list.pop();
            }
            return {this, idx};
        }

        template<typename U>
        PoolHandle<T> create(U&& obj) {
            PoolHandle<T> handle = create();
            properties.get<T>(objects_name)[handle.idx] = std::forward<U>(obj);
            return handle;
        }

        void destroy(const PoolHandle<T> &handle) {
            if (handle.pool && handle.idx < handle.pool->properties.size()) {
                size_t &ref_count = handle.pool->ref_count[handle.idx];
                if (ref_count > 0) {
                    --ref_count;
                }
                if (ref_count == 0) {
                    free_list.push(handle.idx);
                }
            }
        }

    protected:
        friend struct PoolHandle<T>;
        static constexpr const char *ref_count_name = "ref_count";
        static constexpr const char *objects_name = "objects";
        std::queue<size_t> free_list;
    };

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

#endif //ENGINE24_POOL_H
