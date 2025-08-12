//
// Created by alex on 12.08.25.
//

#ifndef ENGINE24_RESOURCESPOINTCLOUD_H
#define ENGINE24_RESOURCESPOINTCLOUD_H

#include "GuiModule.h"
#include "PointCloud.h"
#include "Events/EventsCallbacks.h"

namespace Bcg {
    struct ResourcesPointCloud {
        static void activate();

        static PointCloud load(const std::string &filepath);

        static void on_drop_file(const Events::Callback::Drop &event);
    };
}
#endif //ENGINE24_RESOURCESPOINTCLOUD_H
