//
// Created by alex on 30.10.24.
//
#ifndef ENGINE24_POOL_H
#define ENGINE24_POOL_H

#include "Properties.h"
#include <queue>

namespace Bcg {
    template<typename T>
    struct PoolHandle;


    template<typename T>
    class Pool {
    public:
        Pool() : ref_count(properties.get_or_add<size_t>("ref_count", 0)),
                 objects(properties.get_or_add<T>("objects")),
                 generations(properties.get_or_add<size_t>("generations", 0)) {

        }

        virtual ~Pool() = default;

        PropertyContainer properties;
        Property<size_t> ref_count;
        Property<T> objects;
        Property<size_t> generations;

        template<typename U>
        PoolHandle<T> create_smart_handle(U &&obj) {
            size_t idx;
            if (free_list.empty()) {
                idx = properties.size();
                properties.push_back();
                generations[idx] = 1; // Initialize generation for the new object
            } else {
                idx = free_list.front();
                free_list.pop();
                ++generations[idx];
            }
            objects[idx] = std::forward<U>(obj);
            return {this, idx, generations[idx]};
        }

        void destroy(const PoolHandle<T> &handle) {
            if (handle.pool != this) return; // Ensure the handle belongs to this pool
            if (handle.pool == nullptr) return; // Handle is invalid
            if (handle.idx >= properties.size()) return; // Invalid index

            size_t &r_count = ref_count[handle.idx];
            if (r_count > 0) {
                --r_count;
            }

            if (r_count == 0) {
                objects[handle.idx].~T(); // Call destructor for the object
                objects[handle.idx] = T(); // Reset the object
                free_list.push(handle.idx);
            }
        }

    protected:
        friend struct PoolHandle<T>;
        std::queue<size_t> free_list;
    };
}

#endif //ENGINE24_POOL_H
