//
// Created by alex on 13.06.25.
//

#ifndef ENGINE24_MEMORYPOOL_H
#define ENGINE24_MEMORYPOOL_H

#include <vector>
#include <queue>
#include <limits>
#include "DumbHandle.h"

namespace Bcg {
    template<typename T>
    class MemoryPool {
    public:
        DumbHandle create_handle(T &&object) {
            size_t idx = std::numeric_limits<size_t>::max();
            if (free_list.empty()) {
                idx = pool.size();
                pool.emplace_back(std::move(object));
                generations.push_back(1);
            } else {
                idx = free_list.front();
                free_list.pop();
                new (&pool[idx]) T(std::move(object)); // Placement new to construct the object in the existing memory
            }
            return DumbHandle(idx, generations[idx]);
        }

        bool is_valid(const DumbHandle &handle) const {
            return handle.index < pool.size() && handle.generation == generations[handle.index];
        }

        bool destroy(const DumbHandle &handle) {
            if (!is_valid(handle)) {
                return false; // Invalid handle
            }
            pool[handle.index].~T();
            free_list.push(handle.index); // Add index to free list
            ++generations[handle.index]; // Increment generation count for the recycled object
            return true;
        }

        T *get_object(const DumbHandle &handle) {
            if (!is_valid(handle)) {
                return nullptr; // Invalid handle
            }
            return &pool[handle.index];
        }

        const T *get_object(const DumbHandle &handle) const {
            if (!is_valid(handle)) {
                return nullptr; // Invalid handle
            }
            return &pool[handle.index];
        }

        size_t get_generation(const DumbHandle &handle) const {
            if (!is_valid(handle)) {
                return 0; // Invalid handle
            }
            return generations[handle.index];
        }

    protected:
        std::vector<T> pool;
        std::vector<size_t> generations; // Generation count for each object
        std::queue<size_t> free_list; // Free list for recycled indices
    };
}

#endif //ENGINE24_MEMORYPOOL_H
