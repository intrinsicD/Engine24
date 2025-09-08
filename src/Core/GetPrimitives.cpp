//
// Created by alex on 01.08.24.
//

#include "GetPrimitives.h"
#include "Engine.h"
#include "ModuleMesh.h"
#include "ModulePointCloud.h"
#include "Graph.h"

namespace Bcg {
    Vertices *GetPrimitives::vertices() const {
        if (Engine::has<MeshHandle>(entity_id)) {
            return &Engine::State().get<MeshHandle>(entity_id)->data.vertices;
        }
        if (Engine::has<PointCloudHandle>(entity_id)) {
            return &Engine::State().get<PointCloudHandle>(entity_id)->data.vertices;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).data.vertices;
        }
        if (Engine::has<Graph>(entity_id)) {
            return &Engine::State().get<Graph>(entity_id).data.vertices;
        }
        if (Engine::has<PointCloud>(entity_id)) {
            return &Engine::State().get<PointCloud>(entity_id).data.vertices;
        }
        return nullptr;
    }

    Halfedges *GetPrimitives::halfedges() const {
        if (Engine::has<MeshHandle>(entity_id)) {
            return &Engine::State().get<MeshHandle>(entity_id)->data.halfedges;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).data.halfedges;
        }
        if (Engine::has<Graph>(entity_id)) {
            return &Engine::State().get<Graph>(entity_id).data.halfedges;
        }
        return nullptr;
    }

    Edges *GetPrimitives::edges() const {
        if (Engine::has<MeshHandle>(entity_id)) {
            return &Engine::State().get<MeshHandle>(entity_id)->data.edges;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).data.edges;
        }
        if (Engine::has<Graph>(entity_id)) {
            return &Engine::State().get<Graph>(entity_id).data.edges;
        }
        return nullptr;
    }

    Faces *GetPrimitives::faces() const {
        if (Engine::has<MeshHandle>(entity_id)) {
            return &Engine::State().get<MeshHandle>(entity_id)->data.faces;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).data.faces;
        }
        return nullptr;
    }
}
