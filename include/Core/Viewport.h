//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_VIEWPORT_H
#define ENGINE24_VIEWPORT_H

#include "MatVec.h"

namespace Bcg {
    class Viewport {
    public:
        Viewport() = default;

        Viewport(const Vector<int, 4> &viewport)
                : m_viewport(viewport) {}

        Viewport(int x = 0, int y = 0, int width = 1, int height = 1)
                : m_viewport(x, y, width, height) {}

        // Getters
        Vector<int, 2> get_position() const {
            return {m_viewport[0], m_viewport[1]};
        }

        Vector<int, 2> get_size() const {
            return {m_viewport[2], m_viewport[3]};
        }

        Vector<int, 4> vec() const {
            return m_viewport;
        }

        operator Vector<int, 4>() const {
            return m_viewport;
        }

    private:
        Vector<int, 4> m_viewport = {0, 0, 1, 1}; // x, y, width, height
    };
}

#endif //ENGINE24_VIEWPORT_H
