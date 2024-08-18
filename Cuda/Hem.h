//
// Created by alex on 18.08.24.
//

#ifndef ENGINE24_HEM_H
#define ENGINE24_HEM_H

#include "MatVec.h"
#include <vector>

namespace Bcg {
    struct HemResult {
        std::vector<Vector<float, 3>> means;
        std::vector<Matrix<float, 3, 3>> covs;
        std::vector<float> weights;
        std::vector<Vector<float, 3>> nvars;
    };
}
namespace Bcg::cuda {
    HemResult Hem(const std::vector<Vector<float, 3>> &positions, unsigned int levels = 5, unsigned int k = 12);
}

#endif //ENGINE24_HEM_H
