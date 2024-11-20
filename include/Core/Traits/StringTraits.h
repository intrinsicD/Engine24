//
// Created by alex on 11/5/24.
//

#ifndef STRINGTRAITS_H
#define STRINGTRAITS_H

#include <sstream>

namespace Bcg {
    template<typename T>
    struct StringTraits {
        static std::string ToString(const T &t) {
            std::stringstream ss;
            ss << t;
            return ss.str();
        }
    };

} // namespace Bcg

#endif //STRINGTRAITS_H
