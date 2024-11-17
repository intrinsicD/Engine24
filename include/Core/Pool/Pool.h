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
}

#endif //ENGINE24_POOL_H
