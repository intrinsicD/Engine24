//
// Created by alex on 15.07.24.
//

#include "ModuleAABB.h"
#include "Engine.h"
#include "Entity.h"
#include "imgui.h"
#include "GetPrimitives.h"
#include "EventsEntity.h"
#include "Types.h"
#include "BoundingVolumes.h"

namespace Bcg {
    static void on_cleanup_components(const Events::Entity::CleanupComponents &event) {
        ModuleAABB::cleanup(event.entity_id);
    }

    ModuleAABB::ModuleAABB() : Module("AABB") {

    }

    void ModuleAABB::activate() {
        if (base_activate()) {
            Engine::Dispatcher().sink<Events::Entity::CleanupComponents>().connect<&on_cleanup_components>();
            if (!Engine::Context().find<Pool<AABB> >()) {
                auto &pool = Engine::Context().emplace<Pool<AABB> >();
                pool.create();
            }
        }
    }

    void ModuleAABB::begin_frame() {
    }

    void ModuleAABB::update() {
    }

    void ModuleAABB::end_frame() {
    }

    void ModuleAABB::deactivate() {
        if (base_deactivate()) {
            Engine::Dispatcher().sink<Events::Entity::CleanupComponents>().disconnect<&on_cleanup_components>();
        }
    }

    void ModuleAABB::render() {
    }

    static std::string s_name = "AABB";

    void ModuleAABB::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("Setup {} failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        auto *vertices = GetPrimitives(entity_id).vertices();

        if (!vertices) {
            Log::Warn("Setup {} failed, entity {} has no vertices.", s_name, entity_id);
            return;
        }

        auto positions = vertices->get<PointType>("v:position");
        if (!positions) {
            Log::Warn("Setup {} failed, entity {} has no property {}.", s_name, entity_id, positions.name());
            return;
        }

        auto &pool = Engine::Context().get<Pool<AABB> >();
        auto &bv = Engine::State().get_or_emplace<BoundingVolumes>(entity_id);
        bv.h_aabb = pool.create();
        *bv.h_aabb = AABB::Build(positions.vector().begin(), positions.vector().end());
        Log::Info("Setup {} for entity {}", s_name, entity_id);
    }

    void ModuleAABB::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("Cleanup {} failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        if (!Engine::has<BoundingVolumes>(entity_id) ||
            Engine::State().get<BoundingVolumes>(entity_id).h_aabb.is_valid()) {
            Log::Warn( "Cleanup {} failed, Entity {} does not have an {}. Abort Command", s_name, entity_id, s_name);
            return;
        }

        auto &bv = Engine::State().get<BoundingVolumes>(entity_id);

        if (bv.h_aabb.is_valid()) {
            auto &pool = Engine::Context().get<Pool<AABB> >();
            pool.destroy(bv.h_aabb);
            assert(!bv.h_aabb.is_valid());
        }

        Log::Info("Cleanup {} for entity {}", s_name, entity_id);
    }

    void ModuleAABB::center_and_scale_by_aabb(entt::entity entity_id, const std::string &property_name) {
        if (!Engine::valid(entity_id)) {
            return;
        }

        if (!Engine::has<BoundingVolumes>(entity_id)) {
            return;
        }

        auto &bv = Engine::State().get<BoundingVolumes>(entity_id);
        if (!bv.h_aabb.is_valid()) {
            return;
        }

        auto &aabb = *bv.h_aabb;

        auto *vertices = GetPrimitives(entity_id).vertices();

        if (!vertices) {
            Log::Warn("Center and scale by {} failed, entity {} has no vertices.", s_name, entity_id);
            return;
        }

        auto data = vertices->get<PointType>(property_name);
        if (!data) {
            Log::Warn("Center and scale by {} failed, entity {} has no property {}.", s_name, entity_id, property_name);
            return;
        }

        Vector<float, 3> c = aabb.center();
        float s = glm::compMax(aabb.max - aabb.min);

        for (auto &point: data.vector()) {
            point -= c;
            point /= s;
        }

        aabb.min = (aabb.min - c) / s;
        aabb.max = (aabb.max - c) / s;
        Log::Info("Center and scale by {} for entity {}", s_name, entity_id);
    }

}
