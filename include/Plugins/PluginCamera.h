//
// Created by alex on 08.07.24.
//

#ifndef ENGINE24_PLUGINCAMERA_H
#define ENGINE24_PLUGINCAMERA_H

#include "Plugin.h"
#include "Camera.h"
#include "Command.h"
#include "MatVec.h"

namespace Bcg {
    class PluginCamera : public Plugin {
    public:
        explicit PluginCamera();

        ~PluginCamera() override = default;

        static Camera<float> *setup(entt::entity entity_id);

        static void setup(Camera<float> &camera);

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

    namespace Commands{
        struct CenterCameraAtDistance : public AbstractCommand {
            explicit CenterCameraAtDistance(const Eigen::Vector<float, 3> &center, float distance = 3) :
                    AbstractCommand("CenterCamera"),
                    center(center),
                    distance(distance) {}

            void execute() const override;

            Eigen::Vector<float, 3> center;
            float distance;
        };

        struct FitNearAndFarToDistance : public AbstractCommand {
            explicit FitNearAndFarToDistance(float distance = 3) : AbstractCommand("FitNearAndFarToDistance"),
                                                                   distance(distance) {

            }

            void execute() const override;

            float distance;
        };
    }
}

#endif //ENGINE24_PLUGINCAMERA_H
