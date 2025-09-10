#pragma once

#include "GeometryData.h"
#include "RequireComponent.h"
#include "Graph/GraphInterface.h"
#include "Mesh/MeshInterface.h"
#include "PointCloud/PointCloudInterface.h"

namespace Bcg {
    template<>
    inline HalfedgeMeshInterface &Require<HalfedgeMeshInterface>(entt::entity entity_id, entt::registry &registry) {
        auto &vertices = Require<Vertices>(entity_id, registry);
        auto &halfedges = Require<Halfedges>(entity_id, registry);
        auto &edges = Require<Edges>(entity_id, registry);
        auto &faces = Require<Faces>(entity_id, registry);
        if (!registry.all_of<HalfedgeMeshInterface>(entity_id)) {
            HalfedgeMeshInterface mesh(vertices, halfedges, edges, faces);
            return registry.emplace<HalfedgeMeshInterface>(entity_id, std::move(mesh));
        }
        return registry.get<HalfedgeMeshInterface>(entity_id);
    }

    template<>
    inline GraphInterface &Require<GraphInterface>(entt::entity entity_id, entt::registry &registry) {
        auto &vertices = Require<Vertices>(entity_id, registry);
        auto &halfedges = Require<Halfedges>(entity_id, registry);
        auto &edges = Require<Edges>(entity_id, registry);
        if (!registry.all_of<GraphInterface>(entity_id)) {
            GraphInterface graph(vertices, halfedges, edges);
            return registry.emplace<GraphInterface>(entity_id, std::move(graph));
        }
        return registry.get<GraphInterface>(entity_id);
    }

    template<>
    inline PointCloudInterface &Require<PointCloudInterface>(entt::entity entity_id, entt::registry &registry) {
        auto &vertices = Require<Vertices>(entity_id, registry);
        if (!registry.all_of<PointCloudInterface>(entity_id)) {
            PointCloudInterface pc(vertices);
            return registry.emplace<PointCloudInterface>(entity_id, std::move(pc));
        }
        return registry.get<PointCloudInterface>(entity_id);
    }
}
