//
// Created by alex on 06.06.25.
//

#ifndef ENGINE24_MODULEGRAPHVIEW_H
#define ENGINE24_MODULEGRAPHVIEW_H

#include "Module.h"
#include "GraphView.h"

namespace Bcg{
    class ModuleGraphView : public Module {
    public:
        ModuleGraphView() : Module("ModuleGraphView") {}

        void activate() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;

        static void show_gui(entt::entity entity_id);

        static void show_gui(entt::entity entity_id, GraphView &view);

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        static void set_positions(entt::entity entity_id, const std::string &property_name);

        static void set_colors(entt::entity entity_id, const std::string &property_name);

        static void set_scalarfield(entt::entity entity_id, const std::string &property_name);

        static void set_uniform_color(entt::entity entity_id, const Vector<float, 3> &uniform_color);

        static void set_edges(entt::entity entity_id, const std::vector<Vector<unsigned int, 2>> &edges);
    };
}

#endif //ENGINE24_MODULEGRAPHVIEW_H
