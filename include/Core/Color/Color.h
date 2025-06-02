//
// Created by alex on 23.07.24.
//

#ifndef ENGINE24_COLOR_H
#define ENGINE24_COLOR_H

#include <cstdint>
#include "MatVec.h"

namespace Bcg {
    uint32_t floatColorToUint32(const Vector<float, 4> &rgba);

    void uint32ToFloatColor(uint32_t uicolor, Vector<float, 4> &rgba);
}

#endif //ENGINE24_COLOR_H
