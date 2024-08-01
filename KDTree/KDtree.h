//
// Created by alex on 01.08.24.
//

#ifndef ENGINE24_KDTREE_H
#define ENGINE24_KDTREE_H

#include <vector>
#include "MatVec.h"
#include "nanoflann.hpp"

namespace Bcg {
    struct QueryResult {
        std::vector<size_t> indices;
        std::vector<float> distances;
    };

    class KDTree {
        struct VectorAdapter {
            explicit VectorAdapter(const std::vector<Vector<float, 3>> &points) : points(points) {}

            const std::vector<Vector<float, 3>> &points;

            inline size_t kdtree_get_point_count() const { return points.size(); }

            inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
                return points[idx][dim];
            }

            template<class BBOX>
            bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
        };

        using Type = nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<float, VectorAdapter>,
                VectorAdapter, 3>;
    public:
        KDTree();

        ~KDTree();

        void build(const std::vector<Vector<float, 3>> &positions);

        QueryResult knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const;

        QueryResult radius_query(const Vector<float, 3> &query_point, float radius) const;

        QueryResult closest_query(const Vector<float, 3> &query_point) const;

    private:
        std::unique_ptr<VectorAdapter> dataset;
        std::unique_ptr<Type> index;
    };
}

#endif //ENGINE24_KDTREE_H
