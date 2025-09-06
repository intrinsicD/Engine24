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
        if(Engine::has<MeshHandle>(entity_id)){
            return &Engine::State().get<MeshHandle>(entity_id)->vprops_;
        }
        if(Engine::has<PointCloudHandle>(entity_id)){
            return &Engine::State().get<PointCloudHandle>(entity_id)->data.vertices;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).vprops_;
        }
        if(Engine::has<Graph>(entity_id)){
            return &Engine::State().get<Graph>(entity_id).data.vertices;
        }
        if (Engine::has<PointCloud>(entity_id)) {
            return &Engine::State().get<PointCloud>(entity_id).data.vertices;
        }
        return nullptr;
    }

    Halfedges *GetPrimitives::halfedges() const {
        if(Engine::has<MeshHandle>(entity_id)){
            return &Engine::State().get<MeshHandle>(entity_id)->hprops_;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).hprops_;
        }
        if(Engine::has<Graph>(entity_id)){
            return &Engine::State().get<Graph>(entity_id).data.halfedges;
        }
        return nullptr;
    }

    Edges *GetPrimitives::edges() const {
        if(Engine::has<MeshHandle>(entity_id)){
            return &Engine::State().get<MeshHandle>(entity_id)->eprops_;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).eprops_;
        }
        if(Engine::has<Graph>(entity_id)){
            return &Engine::State().get<Graph>(entity_id).data.edges;
        }
        return nullptr;
    }

    Faces *GetPrimitives::faces() const {
        if(Engine::has<MeshHandle>(entity_id)){
            return &Engine::State().get<MeshHandle>(entity_id)->fprops_;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).fprops_;
        }
        return nullptr;
    }
}