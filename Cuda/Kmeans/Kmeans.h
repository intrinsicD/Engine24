//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_KMEANS_H
#define ENGINE24_KMEANS_H

#include <vector>
#include "MatVec.h"

namespace Bcg{
    struct KMeansResult{
        float error = 0;
        std::vector<unsigned int> labels;
        std::vector<float> distances;
        std::vector<Vector<float, 3>> centroids;
    };


    KMeansResult KMeans(const std::vector<Vector<float, 3>> &positions, unsigned int k, unsigned int iterations = 100);

    KMeansResult KMeans(const std::vector<Vector<float, 3>> &positions, const std::vector<Vector<float, 3>> &init_means, unsigned int iterations = 100);
}

#endif //ENGINE24_KMEANS_H
