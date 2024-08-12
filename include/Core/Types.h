//
// Created by alex on 27.06.24.
//

#ifndef ENGINE24_TYPES_H
#define ENGINE24_TYPES_H

#include <cstdint>
#include "MatVec.h"

namespace Bcg {
    //! Scalar type
#ifdef BCG_SCALAR_TYPE_64
    using Scalar = double;
#else
    using ScalarType = float;
#endif

//! Point type
    using PointType = Vector<ScalarType, 3>;

//! Normal type
    using NormalType = Vector<ScalarType, 3>;

//! Color type
//! \details RGB values in the range of [0,1]
    using ColorType = Vector<ScalarType, 3>;

//! Texture coordinate type
    using TexCoordType = Vector<ScalarType, 2>;

// define index type to be used
#ifdef BCG_INDEX_TYPE_64
    using IndexType = std::uint_least64_t;
#define BCG_MAX_INDEX UINT_LEAST64_MAX
#else
    using IndexType = std::uint_least32_t;
#define BCG_MAX_INDEX UINT_LEAST32_MAX
#endif
}

#endif //ENGINE24_TYPES_H
