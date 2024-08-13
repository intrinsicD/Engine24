//
// Created by alex on 13.08.24.
//

#ifndef ENGINE24_SPATIALQUERYRESULT_H
#define ENGINE24_SPATIALQUERYRESULT_H

#include <vector>
#include <cstddef>

namespace Bcg {
    struct QueryResult {
        std::vector<size_t> indices;
        std::vector<float> distances;
    };
}

#endif //ENGINE24_SPATIALQUERYRESULT_H
