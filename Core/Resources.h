//
// Created by alex on 03.07.24.
//

#ifndef ENGINE24_RESOURCES_H
#define ENGINE24_RESOURCES_H

#include "../Properties.h"
#include "Engine.h"

namespace Bcg {
    template<typename T>
    struct ResourceContainer : public PropertyContainer {
        ResourceContainer() : PropertyContainer() {
            pool = add<T>("Data");
        }

        ~ResourceContainer() override {}

        Property<T> pool;
        std::vector<unsigned int> free_list;
        std::set<unsigned int> used_list;
    };

    template<typename T>
    class Resources {
    public:
        Resources() : container(Engine::Context().find<ResourceContainer<T>>()) {
            if (!container) {
                container = &Engine::Context().emplace<ResourceContainer<T>>();
            }
        }

        std::pair<T &, size_t> push_back() {
            return push_back(T());
        }

        std::pair<T &, size_t> push_back(T &instance) {
            size_t instance_id;
            if (!container->free_list.empty()) {
                instance_id = container->free_list.back();
                container->free_list.pop_back();
            } else {
                instance_id = container->pool.vector().size();
                container->push_back();
            }
            container->pool[instance_id] = instance;
            container->used_list.emplace(instance_id);
            return {container->pool[instance_id], instance_id};
        }

        bool remove(size_t idx) {
            if (container->pool.vector().size() > idx) {
                container->free_list.push_back(idx);
                container->used_list.erase(idx);
                return true;
            }
            return false;
        }

        T &operator[](size_t idx) {
            return container->pool[idx];
        }

        const T &operator[](size_t idx) const {
            return container->pool[idx];
        }

        ResourceContainer<T> *container;
    };
}

#endif //ENGINE24_RESOURCES_H
