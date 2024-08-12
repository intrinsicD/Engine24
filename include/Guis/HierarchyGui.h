//
// Created by alex on 29.07.24.
//

#ifndef ENGINE24_HIERARCHYGUI_H
#define ENGINE24_HIERARCHYGUI_H

#include "Hierarchy.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void Show(const Hierarchy &hierarchy);

    void Edit(entt::entity owner, Hierarchy &hierarchy);

    void ShowHierarchy(entt::entity entity);
}
#endif //ENGINE24_HIERARCHYGUI_H
