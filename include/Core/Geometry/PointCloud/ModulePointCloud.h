//
// Created by alex on 12.08.25.
//

#ifndef ENGINE24_POINTCLOUDMODULE_H
#define ENGINE24_POINTCLOUDMODULE_H

#include "ComponentModule.h"
#include "ComponentHandle.h"
#include "PointCloud.h"
#include "StringTraitsMesh.h"
#include "Events/EventsCallbacks.h"

namespace Bcg {
    class ModulePointCloud : public Module {
    public:
        ModulePointCloud();

        ~ModulePointCloud() override = default;

        void activate() override;

        void deactivate() override;

        // Creation and management -------------------------------------------------------------------------------------

        static void remove(entt::entity entity_id);

        static bool has(entt::entity entity_id);

        static void destroy_entity(entt::entity entity_id);


        // Processing --------------------------------------------------------------------------------------------------

        static PointCloud load_point_cloud(const std::string &filepath);

        static bool save_point_cloud(const std::string &filepath, const PointCloud &pc);

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        // Gui stuff ---------------------------------------------------------------------------------------------------

        void render_menu() override;

        void render_gui() override;


        static void show_gui(const PointCloudInterface &pci);

        static void show_gui(entt::entity entity_id);

        // Events ---------------------------------------------------------------------------------------------------

        void on_drop_file(const Events::Callback::Drop &event);
    };
}

#endif //ENGINE24_POINTCLOUDMODULE_H
