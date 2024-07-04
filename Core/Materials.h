//
// Created by alex on 03.07.24.
//

#ifndef ENGINE24_MATERIALS_H
#define ENGINE24_MATERIALS_H

#include "Plugin.h"
#include "Material.h"
#include "entt/fwd.hpp"

namespace Bcg {
    class Materials : public Plugin {
    public:
        Materials();

        ~Materials() override = default;

        static void setup(entt::entity entity, MeshMaterial &material);

        static void setup(entt::entity entity, GraphMaterial &material);

        static void setup(entt::entity entity, PointCloudMaterial &material);

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

#endif //ENGINE24_MATERIALS_H
