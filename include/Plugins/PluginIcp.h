//
// Created by alex on 13.08.24.
//

#ifndef ENGINE24_PLUGINICP_H
#define ENGINE24_PLUGINICP_H

#include "Plugin.h"
#include "Command.h"

namespace Bcg {


    class PluginIcp : public Plugin {
    public:
        PluginIcp() : Plugin("ICP") {

        }

        ~PluginIcp() override = default;

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
        struct ComputeIcp : public AbstractCommand {
            explicit ComputeIcp(entt::entity source_id, entt::entity target_id, const IcpConfig &config)
                    : AbstractCommand("ComputeIcp"),
                      source_id(source_id),
                      target_id(target_id),
                      config(config) {

            }

            void execute() const override;

            entt::entity source_id;
            entt::entity target_id;
            IcpConfig config;
        };
    }
}

#endif //ENGINE24_PLUGINICP_H
