//
// Created by alex on 23.07.24.
//

#ifndef ENGINE24_COLOR_H
#define ENGINE24_COLOR_H

#include <cstdint>

namespace Bcg {
    uint32_t floatColorToUint32(float r, float g, float b, float a);

    void uint32ToFloatColor(uint32_t color, float &r, float &g, float &b, float &a);
}

#endif //ENGINE24_COLOR_H
