//
// Created by alex on 23.07.24.
//

#ifndef ENGINE24_COLOR_H
#define ENGINE24_COLOR_H

#include <cstdint>
#include "glm/glm.hpp"

namespace Bcg {
    uint32_t floatColorToUint32(const glm::vec4 &rgba);

    void uint32ToFloatColor(uint32_t uicolor, glm::vec4 &rgba);
}

#endif //ENGINE24_COLOR_H
