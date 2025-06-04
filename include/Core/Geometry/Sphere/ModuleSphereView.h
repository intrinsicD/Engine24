//
// Created by alex on 02.08.24.
//

#ifndef ENGINE24_MODULESPHEREVIEW_H
#define ENGINE24_MODULESPHEREVIEW_H

#include "Module.h"
#include "Command.h"
#include "SphereView.h"

namespace Bcg {
    class ModuleSphereView : public Module {
    public:
        ModuleSphereView() : Module("ModuleViewSphere") {}

        void activate() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;

        static void show_gui(entt::entity entity_id, const SphereView &view);

        static void show_gui(entt::entity entity_id);

        static void setup(entt::entity entity_id);

        static void set_position(entt::entity entity_id, const std::string &property_name);

        static void set_normal(entt::entity entity_id, const std::string &property_name);

        static void set_color(entt::entity entity_id, const std::string &property_name);

        static void set_scalarfield(entt::entity entity_id, const std::string &property_name);

        static void set_radius(entt::entity entity_id, const std::string &property_name);

        static void set_uniform_radius(entt::entity entity_id, float radius);

        static void set_uniform_color(entt::entity entity_id, const Vector<float, 3> &color);

        static void set_indices(entt::entity entity_id, const std::vector<unsigned int> &indices);


    };
}

#endif //ENGINE24_MODULESPHEREVIEW_H
