//
// Created by alex on 08.07.24.
//

#ifndef ENGINE24_PLUGINCAMERA_H
#define ENGINE24_PLUGINCAMERA_H

#include "Plugin.h"
#include "Camera.h"

namespace Bcg {
    class PluginCamera : public Plugin {
    public:
        explicit PluginCamera();

        ~PluginCamera() override = default;

        static Camera *setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        static void transform(Camera &camera, const Matrix<float, 4, 4> &transformation);

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

#endif //ENGINE24_PLUGINCAMERA_H
