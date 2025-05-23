//
// Created by alex on 11/5/24.
//

#ifndef DIMTRAITS_H
#define DIMTRAITS_H

namespace Bcg {
    template<typename T>
    struct DimTraits {
        static size_t GetDims(const T &t) {
            return 1;
        }
    };

    template<typename T, int D>
    struct DimTraits<Eigen::Vector<T, D> > {
        static size_t GetDims(const Eigen::Vector<T, D> &t) {
            return D;
        }
    };

    template<typename T>
    struct DimTraits<Eigen::Vector<T, -1> > {
        static size_t GetDims(const Eigen::Vector<T, -1> &t) {
            return t.size();
        }
    };
}

#endif //DIMTRAITS_H
