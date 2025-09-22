#pragma once

#include "glm/glm.hpp"
#include "Macros.h"

namespace Bcg {
    template<typename T>
    struct MatTraits {
    };

    template<typename S, int C, int R, glm::qualifier Q>
    struct MatTraits<glm::mat<C, R, S, Q> > {
        CUDA_HOST_DEVICE

        static glm::mat<C, R, S, Q> transpose(const glm::mat<C, R, S, Q> &m) {
            return glm::transpose(m);
        }

        CUDA_HOST_DEVICE

        static glm::mat<C, R, S, Q> inverse(const glm::mat<C, R, S, Q> &m) {
            return glm::inverse(m);
        }

        CUDA_HOST_DEVICE

        static S determinant(const glm::mat<C, R, S, Q> &m) {
            return glm::determinant(m);
        }

        CUDA_HOST_DEVICE
        static glm::mat<C, R, S, Q> identity() {
            return glm::mat<C, R, S, Q>(1);
        }

        CUDA_HOST_DEVICE
        S trace(const glm::mat<C, R, S, Q> &m) {
            S t(0.0);
            for (int i = 0; i < glm::min(C, R); ++i) {
                t += m[i][i];
            }
            return t;
        }
    };
}
