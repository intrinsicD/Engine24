//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_BVHCUDANEW_H
#define ENGINE24_BVHCUDANEW_H

#include <vector>
#include "MatVec.h"
#include "SpatialHierarchyQuery.h"
#include "entt/fwd.hpp"

namespace Bcg::cuda {
    //TODO test new bvh more and then transition to it!!!
    class BVHCudaNew : public SpatialHierarchyQuery {
    public:
        explicit BVHCudaNew(entt::entity entity_id);

        ~BVHCudaNew() override;

        operator bool() const;

        void build(const std::vector<Vector<float, 3>> &positions) override;

        [[nodiscard]] QueryResult knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const override;

        [[nodiscard]] QueryResult radius_query(const Vector<float, 3> &query_point, float radius) const override;

        [[nodiscard]] QueryResult closest_query(const Vector<float, 3> &query_point) const override;

        [[nodiscard]] std::vector<QueryResult> knn_query_batch(const std::vector<Vector<float, 3>> &query_points, unsigned int num_closest) const override;

        [[nodiscard]] std::vector<QueryResult> radius_query_batch(const std::vector<Vector<float, 3>> &query_points, float radius) const override;

        [[nodiscard]] std::vector<QueryResult> closest_query_batch(const std::vector<Vector<float, 3>> &query_points) const override;

    private:
        entt::entity entity_id;
    };
}
#endif //ENGINE24_BVHCUDANEW_H
