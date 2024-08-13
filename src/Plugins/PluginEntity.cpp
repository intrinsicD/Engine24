//
// Created by alex on 13.08.24.
//

#include "PluginEntity.h"
#include "Engine.h"
#include "EventsEntity.h"
#include "EventsKeys.h"
#include "Picker.h"

namespace Bcg {

    void on_key_delete(const Events::Key::Delete &event) {
        auto &picker = Engine::Context().get<Picked>();
        auto entity_id = picker.entity.id;
        if (!Engine::valid(entity_id)) {
            return;
        }
        Commands::RemoveEntity(entity_id).execute();
    }

    PluginEntity::PluginEntity() : Plugin("Entity") {}

    void PluginEntity::activate() {
        Engine::Dispatcher().sink<Events::Key::Delete>().connect<&on_key_delete>();
        Plugin::activate();
    }

    void PluginEntity::begin_frame() {

    }

    void PluginEntity::update() {

    }

    void PluginEntity::end_frame() {

    }

    void PluginEntity::deactivate() {
        Engine::Dispatcher().sink<Events::Key::Delete>().disconnect<&on_key_delete>();
        Plugin::deactivate();
    }

    void PluginEntity::render_menu() {

    }

    void PluginEntity::render_gui() {

    }

    void PluginEntity::render() {

    }

    void Commands::RemoveEntity::execute() const {
        Engine::Dispatcher().trigger(Events::Entity::CleanupComponents{entity_id});
        Engine::State().destroy(entity_id);
        Log::Info("Entity {} destroyed", entity_id);
    }
}