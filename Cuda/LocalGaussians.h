//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_LOCALGAUSSIANS_H
#define ENGINE24_LOCALGAUSSIANS_H

#include "LocalGaussiansResult.h"

namespace Bcg::cuda{
    LocalGaussiansResult LocalGaussians(const std::vector<Vector<float, 3>> &points, const std::vector<unsigned int> &labels, const std::vector<Vector<float, 3>> &centroids);

    LocalGaussiansResult LocalGaussians(const std::vector<Vector<float, 3>> &points, int k);
}

#endif //ENGINE24_LOCALGAUSSIANS_H
