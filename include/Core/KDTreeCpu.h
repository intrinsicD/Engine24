//
// Created by alex on 01.08.24.
//

#ifndef ENGINE24_KDTREECPU_H
#define ENGINE24_KDTREECPU_H

#include <vector>
#include "MatVec.h"
#include "SpatialQueryResult.h"
#include "nanoflann.hpp"

namespace Bcg {

    class KDTreeCpu {
        struct VectorAdapter {
            explicit VectorAdapter(const std::vector<Vector<float, 3>> &points) : points(points) {}

            const std::vector<Vector<float, 3>> &points;

            [[nodiscard]] inline size_t kdtree_get_point_count() const { return points.size(); }

            [[nodiscard]] inline float kdtree_get_pt(size_t idx, size_t dim) const;

            template<class BBOX>
            bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
        };

        using Type = nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<float, VectorAdapter>,
                VectorAdapter, 3>;
    public:
        KDTreeCpu();

        ~KDTreeCpu();

        void build(const std::vector<Vector<float, 3>> &positions);

        [[nodiscard]] QueryResult knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const;

        [[nodiscard]] QueryResult radius_query(const Vector<float, 3> &query_point, float radius) const;

        [[nodiscard]] QueryResult closest_query(const Vector<float, 3> &query_point) const;

    private:
        std::unique_ptr<VectorAdapter> dataset;
        std::unique_ptr<Type> index;
    };
}

#endif //ENGINE24_KDTREECPU_H
