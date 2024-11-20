//
// Created by alex on 14.11.24.
//

#ifndef ENGINE24_GLMSTRINGTRAITS_H
#define ENGINE24_GLMSTRINGTRAITS_H

#include "StringTraits.h"
#include "GlmToEigen.h"

namespace Bcg{
    // Specialization for glm::vec
    template<typename S, int L, glm::qualifier Q>
    struct StringTraits<glm::vec<L, S, Q> > {
        static std::string ToString(const glm::vec<L, S, Q> &t) {
            std::stringstream ss;
            ss << MapConst(t).transpose();
            return ss.str();
        }
    };

    // Specialization for glm::mat
    template<typename S, int C, int R, glm::qualifier Q>
    struct StringTraits<glm::mat<C, R, S, Q> > {
        static std::string ToString(const glm::mat<C, R, S, Q> &t) {
            std::stringstream ss;
            ss << MapConst(t).transpose();
            return ss.str();
        }
    };
}

#endif //ENGINE24_GLMSTRINGTRAITS_H
