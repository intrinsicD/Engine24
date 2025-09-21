//
// Created by alex on 11/17/24.
//

#ifndef VECTRAITS_H
#define VECTRAITS_H

#include "MatVec.h"
#include "Macros.h"

#include <glm/gtx/component_wise.hpp>

namespace Bcg {
    template<typename T>
    struct VecTraits {
    };

    template<typename S, int L, glm::qualifier Q>
    struct VecTraits<glm::vec<L, S, Q> > {
        CUDA_HOST_DEVICE

        static glm::vec<L, S, Q> cwiseMin(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::min(u, v);
        }

        CUDA_HOST_DEVICE

        static glm::vec<L, S, Q> cwiseMax(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::max(u, v);
        }

        CUDA_HOST_DEVICE

        static glm::vec<L, S, Q>
        clamp(const glm::vec<L, S, Q> &values, const glm::vec<L, S, Q> &mins, const glm::vec<L, S, Q> &maxs) {
            return glm::clamp(values, mins, maxs);
        }

        CUDA_HOST_DEVICE

        static S dot(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::dot(u, v);
        }

        CUDA_HOST_DEVICE

        static S squared_distance(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return squared_length(u - v);
        }

        CUDA_HOST_DEVICE

        static S normalize(const glm::vec<L, S, Q> &u) {
            return glm::normalize(u);
        }

        CUDA_HOST_DEVICE

        static S length(const glm::vec<L, S, Q> &u) {
            return glm::length(u);
        }

        CUDA_HOST_DEVICE

        static S squared_length(const glm::vec<L, S, Q> &u) {
            return glm::dot(u, u);
        }

        CUDA_HOST_DEVICE

        static S prod(const glm::vec<L, S, Q> &u) {
            return glm::compMul(u);
        }

        CUDA_HOST_DEVICE

        static glm::vec<L, bool, Q> lessThan(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::lessThan(u, v);
        }

        CUDA_HOST_DEVICE

        static glm::vec<L, bool, Q> lessThanEqual(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::lessThanEqual(u, v);
        }

        CUDA_HOST_DEVICE

        static glm::vec<L, bool, Q> greaterThan(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::greaterThan(u, v);
        }

        CUDA_HOST_DEVICE

        static glm::vec<L, bool, Q> greaterThanEqual(const glm::vec<L, S, Q> &u, const glm::vec<L, S, Q> &v) {
            return glm::greaterThanEqual(u, v);
        }
    };
}

#endif //VECTRAITS_H
