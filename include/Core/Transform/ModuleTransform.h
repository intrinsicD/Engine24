//
// Created by alex on 15.07.24.
//

#ifndef ENGINE24_MODULETRANSFORM_H
#define ENGINE24_MODULETRANSFORM_H

#include "ComponentModule.h"
#include "entt/fwd.hpp"
#include "TransformUtils.h"
#include "Command.h"

namespace Bcg {
    using TransformHandle = PoolHandle<Transform>;
    using TransformPool = Pool<Transform>;

    class ModuleTransform : public Module {
    public:
        explicit ModuleTransform();

        ~ModuleTransform() override = default;

        void activate() override;

        void deactivate() override;

        static TransformHandle make_handle(const Transform &object);

        static TransformHandle create(entt::entity entity_id, const Transform &object);

        static TransformHandle add(entt::entity entity_id, TransformHandle h_object);

        static void remove(entt::entity entity_id);

        static bool has(entt::entity entity_id);

        static TransformHandle get(entt::entity entity_id);

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void render() override;

        void render_menu() override;

        void render_gui() override;

        static bool show_gui(TransformHandle &h_transform);

        static bool show_gui(Transform &transform);

        static bool show_gui(entt::entity entity_id);

        static Transform *setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        static void set_identity_transform(entt::entity entity_id);
    };
}
#endif //ENGINE24_MODULETRANSFORM_H
