//
// Created by alex on 26.06.24.
//

#ifndef ENGINE24_BUFFERS_H
#define ENGINE24_BUFFERS_H

#include <unordered_map>
#include <string>

namespace Bcg {
    struct Buffers : public std::unordered_map<std::string, unsigned int> {
        using std::unordered_map<std::string, unsigned int>::unordered_map;
    };
}

#endif //ENGINE24_BUFFERS_H
