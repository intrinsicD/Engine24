//
// Created by alex on 05.08.24.
//

#ifndef ENGINE24_VECTORFIELDVIEWGUI_H
#define ENGINE24_VECTORFIELDVIEWGUI_H
#include "VectorfieldView.h"
#include "entt/fwd.hpp"

namespace Bcg::Gui {
    void Show(VectorfieldView &view);

    void ShowVectorfieldView(entt::entity entity_id, VectorfieldView &view);

    void Show(VectorfieldViews &views);

    void ShowVectorfieldViews(entt::entity entity_id);
}
#endif //ENGINE24_VECTORFIELDVIEWGUI_H
