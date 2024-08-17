//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_KMEANSRESULT_H
#define ENGINE24_KMEANSRESULT_H

#include "MatVec.h"

namespace Bcg{
    struct KMeansResult{
        float error = 0;
        std::vector<unsigned int> labels;
        std::vector<float> distances;
        std::vector<Vector<float, 3>> centroids;
        unsigned int max_dist_index;
    };
}

#endif //ENGINE24_KMEANSRESULT_H
