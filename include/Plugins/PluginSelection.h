//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_PLUGINSELECTION_H
#define ENGINE24_PLUGINSELECTION_H

#include "Plugin.h"
#include "Command.h"

namespace Bcg {
    class PluginSelection : public Plugin {
    public:
        PluginSelection();

        ~PluginSelection() = default;

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
        struct MarkPoints : public AbstractCommand {
            MarkPoints(entt::entity entity_id, const std::string &property_name) : AbstractCommand("MarkPoints"),
                                                                                   entity_id(entity_id),
                                                                                   property_name(property_name) {}

            void execute() const override;

            entt::entity entity_id;
            std::string property_name;
        };

        struct EnableVertexSelection : public AbstractCommand {
            EnableVertexSelection(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                    "EnableVertexSelection"),
                                                                                              entity_id(entity_id) {}

            void execute() const override;

            entt::entity entity_id;
        };
    }
}

#endif //ENGINE24_PLUGINSELECTION_H
