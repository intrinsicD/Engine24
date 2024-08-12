//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_PLUGINTRANSFORM_H
#define ENGINE24_PLUGINTRANSFORM_H

#include "Plugin.h"
#include "entt/fwd.hpp"
#include "Transform.h"

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


}
#endif //ENGINE24_PLUGINTRANSFORM_H
