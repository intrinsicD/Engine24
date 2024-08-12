//
// Created by alex on 28.07.24.
//

#ifndef ENGINE24_SHAREDCOMPONENTHANDLE_H
#define ENGINE24_SHAREDCOMPONENTHANDLE_H

#include "Engine.h"

namespace Bcg {
    template<typename T>
    class SharedComponentHandle {
    public:
        SharedComponentHandle() : owner_entity_id(entt::null), component(nullptr) {}

        SharedComponentHandle(entt::entity owner_entity_id) : owner_entity_id(owner_entity_id),
                                                              component(Engine::State().try_get<T>(owner_entity_id)) {}

        SharedComponentHandle(entt::entity owner_entity_id, T *component) : owner_entity_id(owner_entity_id),
                                                                            component(component) {}

        bool try_activate() {
            if (!Engine::valid(owner_entity_id)) return false;
            if (!component) {
                component = Engine::State().try_get<T>(owner_entity_id);
                if(!component){
                    return false;
                }
            }
            return true;
        }

        T &get() const {
            return *component;
        }

        T &operator*() const {
            return *component;
        }

        T *operator->() const {
            return component;
        }

        operator T &() const {
            return *component;
        }

        entt::entity owner() const {
            return owner_entity_id;
        }

        [[nodiscard]] bool is_deleted() const {
            return !Engine::State().all_of<T>(owner_entity_id);
        }

        [[nodiscard]] bool is_valid() const {
            return Engine::valid(owner_entity_id) && component != nullptr;
        }

        explicit operator bool() const {
            return is_valid() && !is_deleted();
        }

        bool operator==(const SharedComponentHandle<T> &other) const {
            return owner_entity_id == other.owner_entity_id && component == other.component;
        }

        bool operator!=(const SharedComponentHandle<T> &other) const {
            return !operator==(other);
        }

    private:
        entt::entity owner_entity_id;
        T *component;
    };
}

#endif //ENGINE24_SHAREDCOMPONENTHANDLE_H
