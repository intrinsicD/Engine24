//
// Created by alex on 13.08.24.
//

#ifndef ENGINE24_PLUGINENTITY_H
#define ENGINE24_PLUGINENTITY_H

#include "Plugin.h"
#include "Command.h"
#include "Entity.h"

namespace Bcg{
    class PluginEntity : public Plugin {
    public:
        PluginEntity();

        ~PluginEntity() override = default;

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };

    namespace Commands{
        struct RemoveEntity : public AbstractCommand {
            explicit RemoveEntity(entt::entity entity_id) : AbstractCommand("RemoveEntity"), entity_id(entity_id) {}

            void execute() const override;

            entt::entity entity_id;
        };
    }
}

#endif //ENGINE24_PLUGINENTITY_H
