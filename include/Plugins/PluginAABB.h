//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_PLUGINAABB_H
#define ENGINE24_PLUGINAABB_H

#include "Plugin.h"
#include "AABBStruct.h"
#include "Command.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class PluginAABB : public Plugin {
    public:
        explicit PluginAABB();

        ~PluginAABB() override = default;

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;
    };

    namespace Commands {
        template<>
        struct Setup<AABB> : public AbstractCommand {
            explicit Setup(entt::entity entity_id) : AbstractCommand("Setup<AABB>"), entity_id(entity_id) {

            }

            void execute() const override;

            entt::entity entity_id;
        };

        template<>
        struct Cleanup<AABB> : public AbstractCommand {
            explicit Cleanup(entt::entity entity_id) : AbstractCommand("Cleanup<AABB>"), entity_id(entity_id) {

            }

            void execute() const override;

            entt::entity entity_id;
        };

        struct CenterAndScaleByAABB : public AbstractCommand {
            explicit CenterAndScaleByAABB(entt::entity entity_id, std::string property_name) : AbstractCommand(
                    "CenterAndScaleByAABB"), entity_id(entity_id), property_name(property_name) {

            }

            void execute() const override;

            entt::entity entity_id;
            std::string property_name;
        };
    }
}

#endif //ENGINE24_PLUGINAABB_H
