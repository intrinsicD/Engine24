//
// Created by alex on 12.08.25.
//

#ifndef ENGINE24_POINTCLOUDMODULE_H
#define ENGINE24_POINTCLOUDMODULE_H

#include "ComponentModule.h"
#include "PointCloud.h"
#include "StringTraitsMesh.h"
#include "Events/EventsCallbacks.h"

namespace Bcg {
    using PointCloudHandle = PoolHandle<PointCloud>;
    using PointCloudPool = Pool<PointCloud>;

    class ModulePointCloud : public Module {
    public:
        ModulePointCloud();

        ~ModulePointCloud() override = default;

        void activate() override;

        void deactivate() override;

        // Creation and management --------------------------------------------------------------------------------------

        static PointCloudHandle make_handle(const PointCloud &pc);

        static PointCloudHandle create(entt::entity entity_id, const PointCloud &pc);

        static PointCloudHandle add(entt::entity entity_id, PointCloudHandle h_pc);

        static void remove(entt::entity entity_id);

        static bool has(entt::entity entity_id);

        static PointCloudHandle get(entt::entity entity_id);

        // Processing ---------------------------------------------------------------------------------------------------

        static PointCloud load_point_cloud(const std::string &filepath);

        static bool save_point_cloud(const std::string &filepath, const PointCloud &pc);

        static void setup(entt::entity entity_id);

        static void cleanup(entt::entity entity_id);

        // Gui stuff ---------------------------------------------------------------------------------------------------

        void render_menu() override;

        void render_gui() override;

        static void show_gui(const PointCloudHandle &h_pc);

        static void show_gui(const PointCloud &pc);

        static void show_gui(entt::entity entity_id);

        // Events ---------------------------------------------------------------------------------------------------

        void on_drop_file(const Events::Callback::Drop &event);
    };
}

#endif //ENGINE24_POINTCLOUDMODULE_H
