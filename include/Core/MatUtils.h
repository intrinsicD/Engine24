//
// Created by alex on 25.10.24.
//

#ifndef ENGINE24_MATUTILS_H
#define ENGINE24_MATUTILS_H

#include "glm/glm.hpp"

namespace Bcg{
    template<typename T, glm::qualifier Q = glm::defaultp>
    T trace(glm::mat<2, 2, T, Q> const &m) {
        return m[0][0] + m[1][1];
    }

    template<typename T, glm::qualifier Q = glm::defaultp>
    T trace(glm::mat<3, 3, T, Q> const &m) {
        return m[0][0] + m[1][1] + m[2][2];
    }

    template<typename T, glm::qualifier Q = glm::defaultp>
    T trace(glm::mat<4, 4, T, Q> const &m) {
        return m[0][0] + m[1][1] + m[2][2] + m[3][3];
    }

    template<typename T, glm::qualifier Q = glm::defaultp>
    T geroshin_radius(const glm::mat<3, 3, T, Q> &m) {
        float radius_col0 = m[0].x + fabsf(m[0].y) + fabsf(m[0].z);
        float radius_col1 = m[1].y + fabsf(m[1].x) + fabsf(m[1].z);
        float radius_col2 = m[2].z + fabsf(m[2].x) + fabsf(m[2].y);
        return fmaxf(radius_col0, fmaxf(radius_col1, radius_col2));
    }

    template<typename T, glm::qualifier Q = glm::defaultp>
    T norm(const glm::mat<3, 3, T, Q> &m) {
        return sqrtf(glm::dot(m[0], m[0]) + glm::dot(m[1], m[1]) + glm::dot(m[2], m[2]));
    }

    template<typename T, glm::qualifier Q = glm::defaultp>
    T max_row_sum(const glm::mat<3, 3, T, Q> &m) {
        float row0_sum = fabsf(m[0].x) + fabsf(m[0].y) + fabsf(m[0].z);
        float row1_sum = fabsf(m[1].x) + fabsf(m[1].y) + fabsf(m[1].z);
        float row2_sum = fabsf(m[2].x) + fabsf(m[2].y) + fabsf(m[2].z);
        return fmaxf(row0_sum, fmaxf(row1_sum, row2_sum));
    }
}
#endif //ENGINE24_MATUTILS_H
