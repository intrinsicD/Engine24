//
// Created by alex on 03.07.24.
//

#ifndef ENGINE24_RESOURCES_H
#define ENGINE24_RESOURCES_H

#include "Properties.h"
#include "Engine.h"
#include "Plugin.h"

namespace Bcg {
    template<typename T>
    struct Resource : public std::pair<T &, size_t> {
        using std::pair<T &, size_t>::pair;
    };

    template<typename T>
    struct ResourceContainer : public PropertyContainer {
        ResourceContainer() : PropertyContainer() {
            pool = add<T>("Data");
        }

        ~ResourceContainer() override = default;

        Property<T> pool;
        std::vector<unsigned int> free_list;
        std::set<unsigned int> used_list;
    };

    template<typename T>
    class Resources {
    public:
        Resources() : container(
                Engine::Context().find<ResourceContainer<T>>() ? Engine::Context().get<ResourceContainer<T>>()
                                                               : Engine::Context().emplace<ResourceContainer<T>>()) {

        }

        Resource<T> create() {
            return push_back(T());
        }

        Resource<T> create_from(const T &object) {
            size_t instance_id;
            if (!container.free_list.empty()) {
                instance_id = container.free_list.back();
                container.free_list.pop_back();
            } else {
                instance_id = container.pool.vector().size();
                container.push_back();
            }
            container.pool[instance_id] = object;
            container.used_list.emplace(instance_id);
            return {container.pool[instance_id], instance_id};
        }

        bool remove(size_t idx) {
            if (container.pool.vector().size() > idx) {
                container.free_list.push_back(idx);
                container.used_list.erase(idx);
                return true;
            }
            return false;
        }

        bool remove(const Resource<T> &resource) {
            return remove(resource.second);
        }

        T &operator[](size_t idx) {
            return container.pool[idx];
        }

        const T &operator[](size_t idx) const {
            return container.pool[idx];
        }

        ResourceContainer<T> &container;
    };

    class ResourcesModule : public Plugin {
    public:
        ResourcesModule();

        ~ResourcesModule() override = default;

        template<typename T>
        static void render_gui(const Resources<T> &container) {

        }

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };
}

#endif //ENGINE24_RESOURCES_H
