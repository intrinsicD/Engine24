//
// Created by alex on 23.07.24.
//

#include "Color.h"

namespace Bcg {
    inline float clamp(float value, float minVal, float maxVal) {
        return (value < minVal ? minVal : (value > maxVal ? maxVal : value));
    }

    uint32_t floatColorToUint32(float r, float g, float b, float a) {
        // Clamp the values to ensure they are within the 0.0 to 1.0 range
        r = clamp(r, 0.0f, 1.0f);
        g = clamp(g, 0.0f, 1.0f);
        b = clamp(b, 0.0f, 1.0f);
        a = clamp(a, 0.0f, 1.0f);

        // Convert to 8-bit per channel
        uint8_t red = static_cast<uint8_t>(r * 255.0f);
        uint8_t green = static_cast<uint8_t>(g * 255.0f);
        uint8_t blue = static_cast<uint8_t>(b * 255.0f);
        uint8_t alpha = static_cast<uint8_t>(a * 255.0f);

        uint32_t color = (red << 24) | (green << 16) | (blue << 8) | alpha;
        return color;
    }

    void uint32ToFloatColor(uint32_t color, float &r, float &g, float &b, float &a) {
        // Extract each component by masking and shifting
        uint8_t alpha = (color >> 0) & 0xFF;
        uint8_t blue = (color >> 8) & 0xFF;
        uint8_t green = (color >> 16) & 0xFF;
        uint8_t red = (color >> 24) & 0xFF;

        // Convert to float in the range [0.0, 1.0]
        r = red / 255.0f;
        g = green / 255.0f;
        b = blue / 255.0f;
        a = alpha / 255.0f;
    }
}