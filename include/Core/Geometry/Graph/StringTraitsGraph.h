//
// Created by alex on 12.08.25.
//

#ifndef ENGINE24_STRINGTRAITSPOINTCLOUD_H
#define ENGINE24_STRINGTRAITSPOINTCLOUD_H

#include "StringTraits.h"
#include "Graph.h"

namespace Bcg{
    template<>
    struct StringTraits<Graph> {
        static std::string ToString(const Graph &t) {
            return "Graph to string not jet implemented";
        }
    };
}

#endif //ENGINE24_STRINGTRAITSPOINTCLOUD_H
