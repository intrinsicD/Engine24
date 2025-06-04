//
// Created by alex on 04.06.25.
//

#ifndef ENGINE24_MODULEPHONGSPLATTINGVIEW_H
#define ENGINE24_MODULEPHONGSPLATTINGVIEW_H

#include "Module.h"
#include "Command.h"
#include "PhongSplattingView.h"

namespace Bcg {
    class ModulePhongSplattingView : public Module {
    public:
        ModulePhongSplattingView() : Module("ModulePhongSplattingView") {}

        void activate() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;

        static void show_gui(entt::entity entity_id, PhongSplattingView &view);

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
#endif //ENGINE24_MODULEPHONGSPLATTINGVIEW_H
