//
// Created by alex on 05.06.25.
//

#ifndef ENGINE24_SPATIALHIERARCHYQUERY_H
#define ENGINE24_SPATIALHIERARCHYQUERY_H

#include "SpatialQueryResult.h"
#include "MatVec.h"

namespace Bcg {
    class SpatialHierarchyQuery {
    public:
        virtual ~SpatialHierarchyQuery() = default;

        virtual void build(const std::vector<Vector<float, 3>> &positions) = 0;

        [[nodiscard]] virtual QueryResult knn_query(const Vector<float, 3> &query_point,
                                                    unsigned int num_closest) const = 0;

        [[nodiscard]] virtual QueryResult radius_query(const Vector<float, 3> &query_point, float radius) const = 0;

        [[nodiscard]] virtual QueryResult closest_query(const Vector<float, 3> &query_point) const = 0;

        [[nodiscard]] virtual std::vector<QueryResult>
        knn_query_batch(const std::vector<Vector<float, 3>> &query_points, unsigned int num_closest) const = 0;

        [[nodiscard]] virtual std::vector<QueryResult>
        radius_query_batch(const std::vector<Vector<float, 3>> &query_points, float radius) const = 0;

        [[nodiscard]] virtual std::vector<QueryResult>
        closest_query_batch(const std::vector<Vector<float, 3>> &query_points) const = 0;
    };
}

#endif //ENGINE24_SPATIALHIERARCHYQUERY_H
