//
// Created by alex on 04.11.24.
//

#ifndef ENGINE24_KDTREESAMPLING_H
#define ENGINE24_KDTREESAMPLING_H

#include <vector>
#include "MatVec.h"

namespace Bcg {
    struct SamplingResult {
        const std::vector<Vector<float, 3>> points;
        std::vector<unsigned int> sample_indices;
        unsigned int level;
    };

    SamplingResult KDTreeSampling(const std::vector<Vector<float, 3>> &points, unsigned int n_level);
}

#endif //ENGINE24_KDTREESAMPLING_H
