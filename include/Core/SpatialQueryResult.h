//
// Created by alex on 13.08.24.
//

#ifndef ENGINE24_SPATIALQUERYRESULT_H
#define ENGINE24_SPATIALQUERYRESULT_H

#include <vector>
#include <cstddef>
#include "Eigen/Core"

namespace Bcg {
    struct QueryResult {
        std::vector<size_t> indices;
        std::vector<float> distances;


        QueryResult() = default;

        explicit QueryResult(size_t size) : indices(size), distances(size) {

        }

        void resize(size_t size) {
            indices.resize(size);
            distances.resize(size);
        }

        void reserve(size_t size) {
            indices.reserve(size);
            distances.reserve(size);
        }

        [[nodiscard]] size_t size() const {
            return indices.size();
        }

        [[nodiscard]] bool empty() const {
            return indices.size() == 0;
        }
    };
}

#endif //ENGINE24_SPATIALQUERYRESULT_H
