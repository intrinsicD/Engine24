//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_KDTREECUDA_H
#define ENGINE24_KDTREECUDA_H

#include "MatVec.h"

namespace Bcg{
    struct QueryResult {
        std::vector<size_t> indices;
        std::vector<float> distances;
    };

    struct KDNode {
        float split_value;
        int left, right, index;
    };

    class KDTree {
    public:
        KDTree();

        ~KDTree();

        void build(const std::vector<Vector<float, 3>> &positions);

        [[nodiscard]] QueryResult knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const;

        [[nodiscard]] QueryResult radius_query(const Vector<float, 3> &query_point, float radius) const;

        [[nodiscard]] QueryResult closest_query(const Vector<float, 3> &query_point) const;
    private:
        KDNode *index;
    };
}

#endif //ENGINE24_KDTREECUDA_H
