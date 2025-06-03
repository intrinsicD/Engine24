//
// Created by alex on 26.11.24.
//

#ifndef ENGINE24_MODULEMESH_H
#define ENGINE24_MODULEMESH_H

#include "ComponentModule.h"
#include "SurfaceMesh.h"
#include "Events/EventsCallbacks.h"

namespace Bcg {
    using MeshHandle = PoolHandle<SurfaceMesh>;
    using MeshPool = Pool<SurfaceMesh>;

    class ModuleMesh : public Module {
    public:
        ModuleMesh();

        ~ModuleMesh() override = default;

        void activate() override;

        void deactivate() override;

        // Creation and management --------------------------------------------------------------------------------------

        static MeshHandle make_handle(const SurfaceMesh &mesh);

        static MeshHandle create(entt::entity entity_id, const SurfaceMesh &mesh);

        static MeshHandle add(entt::entity entity_id, MeshHandle h_mesh);

        static void remove(entt::entity entity_id);

        static bool has(entt::entity entity_id);

        static MeshHandle get(entt::entity entity_id);

        // Processing ---------------------------------------------------------------------------------------------------

        static SurfaceMesh load_mesh(const std::string &filepath);

        static bool save_mesh(const std::string &filepath, const SurfaceMesh &mesh);

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        // Gui stuff ---------------------------------------------------------------------------------------------------

        void render_menu() override;

        void render_gui() override;

        static void show_gui(const MeshHandle &h_mesh);

        static void show_gui(const SurfaceMesh &mesh);

        static void show_gui(entt::entity entity_id);

        // Events ---------------------------------------------------------------------------------------------------

        void on_drop_file(const Events::Callback::Drop &event);
    };
}

#endif //ENGINE24_MODULEMESH_H
