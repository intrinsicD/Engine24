//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_LOCALGAUSSIANS_H
#define ENGINE24_LOCALGAUSSIANS_H

#include "LocalGaussiansResult.h"

namespace Bcg{
    LocalGaussiansResult LocalGaussians(const std::vector<Vector<float, 3>> &points, const std::vector<unsigned int> &labels, const std::vector<Vector<float, 3>> &centroids, const std::vector<float> &distances, const float sigma, const float epsilon, const unsigned int max_iter);
}

#endif //ENGINE24_LOCALGAUSSIANS_H
