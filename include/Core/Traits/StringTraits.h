//
// Created by alex on 11/5/24.
//

#ifndef STRINGTRAITS_H
#define STRINGTRAITS_H

#include <sstream>
#include "GlmToEigen.h"

namespace Bcg {
    template<typename T>
    struct StringTraits {
        static std::string ToString(const T &t) {
            std::stringstream ss;
            ss << t;
            return ss.str();
        }
    };

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
} // namespace Bcg

#endif //STRINGTRAITS_H
