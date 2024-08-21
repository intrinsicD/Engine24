//
// Created by alex on 20.08.24.
//

#ifndef ENGINE24_FUNCTORS_CUH
#define ENGINE24_FUNCTORS_CUH

namespace Bcg::cuda {
    template<typename Source, typename Target>
    struct transform;

    template<typename Object>
    struct merge;

    template<typename Lhs, typename Rhs>
    struct distance;

    template<typename Object>
    struct max;

    template<typename Object>
    struct min;

    template<typename Value>
    struct clamp;

    template<>
    struct clamp<float> {
        __device__ __host__
        inline float operator()(const float &v, const float &min, const float &max) const noexcept {
            return ::fmaxf(min, ::fminf(v, max));
        }
    };

    template<typename Scalar>
    struct data_container {
        Scalar data;
    };
}

#endif //ENGINE24_FUNCTORS_CUH
