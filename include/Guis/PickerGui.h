//
// Created by alex on 16.07.24.
//

#ifndef ENGINE24_PICKERGUI_H
#define ENGINE24_PICKERGUI_H

#include "Picker.h"

namespace Bcg {
    namespace Gui {
        void Show(Picked &picked);

        void Show(Picked::Entity &entity);
    };
}

#endif //ENGINE24_PICKERGUI_H
