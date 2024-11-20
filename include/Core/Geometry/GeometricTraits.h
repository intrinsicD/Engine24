//
// Created by alex on 20.11.24.
//

#ifndef ENGINE24_GEOMETRICTRAITS_H
#define ENGINE24_GEOMETRICTRAITS_H

namespace Bcg{
    template<typename Object1, typename Object2>
    struct ClosestPointTraits;

    template<typename Object1, typename Object2>
    struct DistanceTraits;

    template<typename Object1, typename Object2>
    struct SquaredDistanceTraits;

    template<typename Object1, typename Object2>
    struct ContainsTraits;

    template<typename Object1, typename Object2>
    struct IntersectsTraits;

    template<typename Object1, typename Object2>
    struct IntersectionTraits;

    template<typename Object1, typename Object2>
    struct GetterTraits;
}

#endif //ENGINE24_GEOMETRICTRAITS_H
