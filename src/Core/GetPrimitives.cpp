//
// Created by alex on 01.08.24.
//

#include "GetPrimitives.h"
#include "Engine.h"
#include "ModuleMesh.h"
#include "PointCloud.h"

namespace Bcg {
    Vertices *GetPrimitives::vertices() const {
        if(Engine::has<MeshHandle>(entity_id)){
            return &Engine::State().get<MeshHandle>(entity_id)->vprops_;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).vprops_;
        } /*else if(Engine::has<Graph>(entity_id)){
            return &Engine::State().get<Graph>(entity_id).vprops_;
        } */else if (Engine::has<PointCloud>(entity_id)) {
            return &Engine::State().get<PointCloud>(entity_id).vprops_;
        }
        return nullptr;
    }

    HalfEdges *GetPrimitives::halfedges() const {
        if(Engine::has<MeshHandle>(entity_id)){
            return &Engine::State().get<MeshHandle>(entity_id)->hprops_;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).hprops_;
        } /* else if(Engine::has<Graph>(entity_id)){
            return &Engine::State().get<Graph>(entity_id).hprops_;
        }*/
        return nullptr;
    }

    Edges *GetPrimitives::edges() const {
        if(Engine::has<MeshHandle>(entity_id)){
            return &Engine::State().get<MeshHandle>(entity_id)->eprops_;
        }
        if (Engine::has<SurfaceMesh>(entity_id)) {
            return &Engine::State().get<SurfaceMesh>(entity_id).eprops_;
        }/* else if(Engine::has<Graph>(entity_id)){
            return &Engine::State().get<Graph>(entity_id).eprops_;
        }*/
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