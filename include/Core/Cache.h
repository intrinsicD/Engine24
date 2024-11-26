//
// Created by alex on 26.11.24.
//

#ifndef ENGINE24_CACHE_H
#define ENGINE24_CACHE_H

#include <unordered_map>
#include <string>

namespace Bcg {
    template<typename T>
    using Cache = std::unordered_map<std::string, T>;
}

#endif //ENGINE24_CACHE_H
