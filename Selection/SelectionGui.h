//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_SELECTIONGUI_H
#define ENGINE24_SELECTIONGUI_H

#include "Selection.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui{
    void ShowSelection(entt::entity entity_id);

    void Show(Selection &selection);
}

#endif //ENGINE24_SELECTIONGUI_H
