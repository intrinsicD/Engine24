//
// Created by alex on 15.07.24.
//

#include "ModuleTransform.h"
#include "TransformUtils.h"
#include "Engine.h"
#include "Entity.h"
#include "PluginHierarchy.h"
#include "Pool.h"


namespace Bcg {

    ModuleTransform::ModuleTransform() : Module("Transform") {}

    void ModuleTransform::activate() {
        if (base_activate()) {
            if (!Engine::Context().find<TransformPool>()) {
                Engine::Context().emplace<TransformPool>();
            }
        }
    }

    void ModuleTransform::begin_frame() {

    }

    void ModuleTransform::update() {

    }

    void ModuleTransform::end_frame() {

    }

    void ModuleTransform::deactivate() {
        if (base_deactivate()) {
            if (Engine::Context().find<TransformPool>()) {
                Engine::Context().erase<TransformPool>();
            }
        }
    }

    TransformHandle ModuleTransform::make_handle(const Transform &object){
        auto &pool = Engine::Context().get<TransformPool>();
        return pool.create(object);
    }

    TransformHandle ModuleTransform::create(entt::entity entity_id, const Transform &object){
        auto handle = make_handle(object);
        return add(entity_id, handle);
    }

    TransformHandle ModuleTransform::add(entt::entity entity_id, TransformHandle h_object){
        return Engine::State().get_or_emplace<TransformHandle>(entity_id, h_object);
    }

    void ModuleTransform::remove(entt::entity entity_id){
        Engine::State().remove<TransformHandle>(entity_id);
    }

    bool ModuleTransform::has(entt::entity entity_id){
        return Engine::State().all_of<TransformHandle>(entity_id);
    }

    TransformHandle ModuleTransform::get(entt::entity entity_id){
        return Engine::State().get<TransformHandle>(entity_id);
    }

    void ModuleTransform::render() {

    }

    Transform *ModuleTransform::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return nullptr; }
        if (Engine::has<Transform>(entity_id)) { return &Engine::State().get<Transform>(entity_id); }

        Log::Info("Transform setup for entity: {}", entity_id);
        return &Engine::State().emplace<Transform>(entity_id, Transform());
    }

    void ModuleTransform::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return; }
        if (!Engine::has<Transform>(entity_id)) { return; }

        Engine::State().remove<Transform>(entity_id);
        Log::Info("Transform cleanup for entity: {}", entity_id);
    }

    void ModuleTransform::set_identity_transform(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return; }
        if (!Engine::has<Transform>(entity_id)) { return; }
        Engine::State().get<Transform>(entity_id).set_local(glm::mat4(1.0f));

        PluginHierarchy::mark_transforms_dirty(entity_id);
    }
}