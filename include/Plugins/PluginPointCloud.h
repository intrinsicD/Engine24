//
// Created by alex on 30.07.24.
//

#ifndef ENGINE24_PLUGINPOINTCLOUD_H
#define ENGINE24_PLUGINPOINTCLOUD_H


#include "Plugin.h"
#include "PointCloud.h"
#include "Command.h"


namespace Bcg {
    class PluginPointCloud : public Plugin {
    public:
        PluginPointCloud();

        ~PluginPointCloud() override = default;

        static PointCloud load(const std::string &path);

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
        struct Load<PointCloud> : public AbstractCommand {
            explicit Load(entt::entity entity_id, const std::string &filepath) : AbstractCommand(
                    "Load<PointCloud>"),
                                                                                 entity_id(entity_id),
                                                                                 filepath(filepath) {}

            void execute() const override;

            entt::entity entity_id;
            std::string filepath;
        };

        template<>
        struct Setup<PointCloud> : public AbstractCommand {
            explicit Setup<PointCloud>(entt::entity entity_id) : AbstractCommand("Setup<PointCloud>"),
                                                               entity_id(entity_id) {}

            void execute() const override;

            entt::entity entity_id;
        };

        template<>
        struct Cleanup<PointCloud> : public AbstractCommand {
            explicit Cleanup<PointCloud>(entt::entity entity_id) : AbstractCommand("Cleanup<PointCloud>"),
                                                                 entity_id(entity_id) {}

            void execute() const override;

            entt::entity entity_id;
        };



        struct ComputePointCloudLocalPcasKnn : public AbstractCommand {
            explicit ComputePointCloudLocalPcasKnn(entt::entity entity_id, int num_closest) : AbstractCommand(
                    "ComputePointCloudLocalPcasKnn"),
                                                                                              entity_id(entity_id),
                                                                                              num_closest(
                                                                                                      num_closest) {}

            void execute() const override;

            entt::entity entity_id;
            int num_closest;
        };

        struct ComputeKMeans : public AbstractCommand {
            explicit ComputeKMeans(entt::entity entity_id, int k, unsigned int iterations = 100) : AbstractCommand(
                    "ComputeKMeans"),
                                                                                                   entity_id(entity_id),
                                                                                                   k(k),
                                                                                                   iterations(
                                                                                                           iterations) {}

            void execute() const override;

            entt::entity entity_id;
            int k;
            unsigned int iterations;
        };
    }
}
#endif //ENGINE24_PLUGINPOINTCLOUD_H
