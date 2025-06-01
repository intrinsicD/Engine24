//
// Created by alex on 5/28/25.
//

#ifndef MESHRESOURCES_H
#define MESHRESOURCES_H

#include "GuiModule.h"
#include "SurfaceMesh.h"
#include "Events/EventsCallbacks.h"

namespace Bcg {
    struct MeshResources {
        static void activate();

        static SurfaceMesh load(const std::string &filepath);

        static void on_drop_file(const Events::Callback::Drop &event);
    };
}

#endif //MESHRESOURCES_H
