//
// Created by alex on 12.11.24.
//

#ifndef ENGINE24_POOLGUI_H
#define ENGINE24_POOLGUI_H

#include "Pool.h"
#include "PropertiesGui.h"
#include "imgui.h"

namespace Bcg::Gui {
    template<typename T>
    void ShowPool(Pool<T> &pool){
        Show("PoolProperties",pool.properties);
    }

}

#endif //ENGINE24_POOLGUI_H
