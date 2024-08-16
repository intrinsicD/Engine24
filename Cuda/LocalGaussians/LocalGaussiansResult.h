//
// Created by alex on 16.08.24.
//

#ifndef ENGINE24_LOCALGAUSSIANSRESULT_H
#define ENGINE24_LOCALGAUSSIANSRESULT_H

#include "MatVec.h"

namespace Bcg{
    struct LocalGaussiansResult{
        std::vector<Vector<float, 3>> means;
        std::vector<Matrix<float, 3, 3>> covs;
    };
}

#endif //ENGINE24_LOCALGAUSSIANSRESULT_H
