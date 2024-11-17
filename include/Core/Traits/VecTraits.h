//
// Created by alex on 11/17/24.
//

#ifndef VECTRAITS_H
#define VECTRAITS_H

#include <GlmToEigen.h>

#include "MatVec.h"

namespace Bcg {
    template<typename T>
    struct VecTraits {
    };

    template<typename S, int L, glm::qualifier Q>
    struct VecTraits<glm::vec<L, S, Q> > {
        static glm::vec<L, S, Q> cwiseMin(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::min(u, v);
        }

        static glm::vec<L, S, Q> cwiseMax(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::max(u, v);
        }

        static S prod(const glm::vec<L, S, Q> &u) {
            return glm::compMul(u);
        }

        static glm::vec<L, bool, Q> lessThan(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::lessThan(u, v);
        }

        static glm::vec<L, bool, Q> lessThanEqual(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::lessThanEqual(u, v);
        }

        static glm::vec<L, bool, Q> greaterThan(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::greaterThan(u, v);
        }

        static glm::vec<L, bool, Q> greaterThanEqual(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::greaterThanEqual(u, v);
        }

    };
}

#endif //VECTRAITS_H
