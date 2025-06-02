//
// Created by alex on 26.11.24.
//

#ifndef ENGINE24_COMPONENTMODULE_H
#define ENGINE24_COMPONENTMODULE_H

#include "Module.h"
#include "PoolHandle.h"
#include "Pool.h"
#include "entt/fwd.hpp"

namespace Bcg {
    template<typename T>
    class ComponentModule : public Module {
    public:
        explicit ComponentModule(const std::string &name) : Module(name) {}

        ~ComponentModule() override = default;

        virtual PoolHandle <T> make_handle(const T &object) = 0;

        virtual PoolHandle<T> create(entt::entity entity_id, const T &object) = 0;

        virtual PoolHandle<T> add(entt::entity entity_id, PoolHandle<T> h_object) = 0;

        virtual void remove(entt::entity entity_id) = 0;

        virtual bool has(entt::entity entity_id) = 0;

        virtual PoolHandle<T> get(entt::entity entity_id) = 0;
    };
}

#endif //ENGINE24_GEOMETRYMODULE_H
