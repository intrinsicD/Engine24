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
/*        std::vector<size_t> indices;
        std::vector<float> distances;*/

        Eigen::Matrix<size_t, -1, -1> indices;
        Eigen::Matrix<float, -1, -1> distances;


        bool empty() const {
            return indices.size() == 0;
        }
    };
}

#endif //ENGINE24_SPATIALQUERYRESULT_H
