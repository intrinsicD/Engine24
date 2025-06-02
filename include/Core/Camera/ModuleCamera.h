//
// Created by alex on 08.07.24.
//

#ifndef ENGINE24_ModuleCamera_H
#define ENGINE24_ModuleCamera_H

#include "Plugin.h"
#include "Camera.h"
#include "Command.h"
#include "MatVec.h"

namespace Bcg {
    class ModuleCamera : public Module {
    public:
        explicit ModuleCamera();

        ~ModuleCamera() override = default;

        static Camera *setup(entt::entity entity_id);

        static void setup(Camera &camera);

        static void cleanup(entt::entity entity_id);

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render() override;

        static void center_camera_at_distance(const Vector<float, 3> &center, float distance = 3);

        static void fit_near_and_far_to_distance(float distance = 3);
    };
}

#endif //ENGINE24_ModuleCamera_H
