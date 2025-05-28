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

    class MeshModule : public ComponentModule<SurfaceMesh> {
    public:
        MeshModule();

        ~MeshModule() override = default;

        void activate() override;

        void deactivate() override;

        MeshHandle make_handle(const SurfaceMesh &mesh) override;

        MeshHandle create(entt::entity entity_id, const SurfaceMesh &mesh) override;

        MeshHandle add(entt::entity entity_id, MeshHandle h_mesh) override;

        void remove(entt::entity entity_id) override;

        bool has(entt::entity entity_id) override;

        MeshHandle get(entt::entity entity_id) override;

        MeshHandle load_mesh(const std::string &filepath);

        //TODO move the following into a plugin or another module or a system or what ever i should call it?

        VertexProperty<Vector<float, 3>> compute_vertex_normals(SurfaceMesh &mesh);

        FaceProperty<Vector<float, 3>> compute_face_normals(SurfaceMesh &mesh);

        FaceProperty<Vector<float, 3>> compute_face_centers(SurfaceMesh &mesh);

        EdgeProperty<float> compute_edge_lengths(SurfaceMesh &mesh);

        size_t get_num_connected_components(SurfaceMesh &mesh);

        std::vector<SurfaceMesh> split_connected_components(SurfaceMesh &mesh);

        SurfaceMesh merge_meshes(const std::vector<SurfaceMesh> &meshes);
    };
}

#endif //ENGINE24_MESHMODULE_H
