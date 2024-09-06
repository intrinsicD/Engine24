//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_PLUGINTRANSFORM_H
#define ENGINE24_PLUGINTRANSFORM_H

#include "Plugin.h"
#include "entt/fwd.hpp"
#include "Transform.h"
#include "Command.h"

namespace Bcg {
    class PluginTransform : public Plugin {
    public:
        explicit PluginTransform();

        ~PluginTransform() override = default;

        static Transform *setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

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
        struct Setup<Transform> : public AbstractCommand {
            explicit Setup(entt::entity entity_id) : AbstractCommand("Setup<Transform>"), entity_id(entity_id) {

            }

            void execute() const override;

            entt::entity entity_id;
        };

        template<>
        struct Cleanup<Transform> : public AbstractCommand {
            explicit Cleanup(entt::entity entity_id) : AbstractCommand("Cleanup<Transform>"), entity_id(entity_id) {

            }

            void execute() const override;

            entt::entity entity_id;
        };

        struct SetIdentityTransform : public AbstractCommand {
            explicit SetIdentityTransform(entt::entity entity_id) : AbstractCommand("SetIdentityTransform"), entity_id(entity_id) {

            }

            void execute() const override;

            entt::entity entity_id;
        };


    }
}
#endif //ENGINE24_PLUGINTRANSFORM_H
