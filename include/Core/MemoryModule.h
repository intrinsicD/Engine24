//
// Created by alex on 25.11.24.
//

#ifndef MEMORYMODULE_H
#define MEMORYMODULE_H

#include "Pool.h"
#include "Engine.h"

namespace Bcg {
    class MemoryModule {
    public:
        MemoryModule() = default;

        ~MemoryModule() = default;

        template<typename T>
        static Pool<T> &request_pool() {
            if (!Engine::Context().find<Pool<T>>()) {
                return Engine::Context().emplace<Pool<T>>();
            }
            return Engine::Context().get<Pool<T>>();
        }
    };
}

#endif //MEMORYMODULE_H
