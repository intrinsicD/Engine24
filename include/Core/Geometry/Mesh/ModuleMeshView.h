//
// Created by alex on 04.08.24.
//

#ifndef ENGINE24_MODULEMESHVIEW_H
#define ENGINE24_MODULEMESHVIEW_H

#include "Plugin.h"
#include "Command.h"
#include "MeshView.h"
#include "MatVec.h"

namespace Bcg {
    class ModuleMeshView : public GuiModule {
    public:
        ModuleMeshView();

        ~ModuleMeshView() override = default;

        void activate() override;

        void deactivate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void render() override;

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        static void set_positions(entt::entity entity_id, const std::string &property_name = "v:position");

        static void set_normals(entt::entity entity_id, const std::string &property_name = "v:normal");

        static void set_colors(entt::entity entity_id, const std::string &property_name = "v:color");

        static void set_uniform_color(entt::entity entity_id, const Vector<float, 3> &data);

        static void set_scalarfield(entt::entity entity_id, const std::string &property_name = "v:scalarfield");

        static void set_triangles(entt::entity entity_id, std::vector<Vector<unsigned int, 3>> &tris);

        // Gui stuff --------------------------------------------------------------------------------------

        void render_menu() override;

        void render_gui() override;

        static void show_gui(entt::entity entity_id, MeshView &view);

        static void show_gui(entt::entity entity_id);
    };
}
#endif //ENGINE24_MODULEMESHVIEW_H
