//
// Created by alex on 30.07.24.
//

#ifndef ENGINE24_POINTCLOUDGUI_H
#define ENGINE24_POINTCLOUDGUI_H

#include "PointCloud.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void ShowLoadPointCloud();

    void ShowPointCloud(entt::entity entity_id);

    void Show(PointCloud &pc);
}
#endif //ENGINE24_POINTCLOUDGUI_H
