//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_CUDACOMMON_CUH
#define ENGINE24_CUDACOMMON_CUH

#include <string>

namespace Bcg::cuda{
    bool CudaCheckErrorAndSync(const std::string &func_name = "");
}

#endif //ENGINE24_CUDACOMMON_CUH
