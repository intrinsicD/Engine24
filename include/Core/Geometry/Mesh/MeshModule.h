//
// Created by alex on 26.11.24.
//

#ifndef ENGINE24_MESHMODULE_H
#define ENGINE24_MESHMODULE_H

#include "ComponentModule.h"
#include "SurfaceMesh.h"

namespace Bcg {
    using MeshHandle = PoolHandle<SurfaceMesh>;
    using MeshPool = Pool<SurfaceMesh>;

    class MeshModule : public Module {
    public:
        MeshModule();

        ~MeshModule() override = default;

        void activate() override;

        void deactivate() override;

        static MeshHandle make_handle(const SurfaceMesh &mesh);

        static MeshHandle create(entt::entity entity_id, const SurfaceMesh &mesh);

        static MeshHandle add(entt::entity entity_id, MeshHandle h_mesh);

        static void remove(entt::entity entity_id);

        static bool has(entt::entity entity_id);

        static MeshHandle get(entt::entity entity_id);

        static SurfaceMesh load_mesh(const std::string &filepath);

        static bool save_mesh(const std::string &filepath, const SurfaceMesh &mesh);
    };
}

#endif //ENGINE24_MESHMODULE_H
