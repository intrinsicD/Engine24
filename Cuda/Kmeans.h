//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_KMEANS_H
#define ENGINE24_KMEANS_H

#include "KmeansResult.h"

namespace Bcg::cuda{
    KMeansResult KMeans(const std::vector<Vector<float, 3>> &positions, unsigned int k, unsigned int iterations = 100);

    KMeansResult KMeans(const std::vector<Vector<float, 3>> &positions, const std::vector<Vector<float, 3>> &init_means, unsigned int iterations = 100);

    KMeansResult HierarchicalKMeans(const std::vector<Vector<float, 3>> &positions, unsigned int k, unsigned int iterations = 100);
}

#endif //ENGINE24_KMEANS_H
