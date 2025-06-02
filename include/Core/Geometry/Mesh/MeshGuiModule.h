//
// Created by alex on 26.11.24.
//

#ifndef ENGINE24_MESHGUIMODULE_H
#define ENGINE24_MESHGUIMODULE_H

#include "GuiModule.h"
#include "SurfaceMesh.h"
#include "PoolHandle.h"
#include "MeshComponent.h"
#include "EventsGui.h"
#include "entt/fwd.hpp"

namespace Bcg{
    class MeshGuiModule : public GuiModule {
    public:
        MeshGuiModule();

        ~MeshGuiModule() override = default;

        void activate() override;

        void deactivate() override;

        static void render_filedialog();

        static void render(const MeshComponent &meshes);

        static void render(const PoolHandle<SurfaceMesh> &handle);

        static void render(const SurfaceMesh &mesh);

        static void render(Pool<SurfaceMesh> &pool);

        static void render(entt::entity entity_id);

        void render_menu() override;

        void render_gui() override;
    };
}

#endif //ENGINE24_MESHGUIMODULE_H
