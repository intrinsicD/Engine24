//
// Created by alex on 08.08.24.
//

#ifndef ENGINE24_KDTREECUDA_H
#define ENGINE24_KDTREECUDA_H

#include <vector>
#include "MatVec.h"
#include "SpatialQueryResult.h"
#include "entt/fwd.hpp"

namespace Bcg::cuda {
    class KDTreeCuda {
    public:
        explicit KDTreeCuda(entt::entity entity_id);

        ~KDTreeCuda();

        operator bool() const;

        void build(const std::vector<Vector<float, 3>> &positions);

        [[nodiscard]] QueryResult knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const;

        [[nodiscard]] QueryResult radius_query(const Vector<float, 3> &query_point, float radius) const;

        [[nodiscard]] QueryResult closest_query(const Vector<float, 3> &query_point) const;

        std::vector<std::uint32_t> get_samples(unsigned int level) const;

        unsigned int compute_num_levels() const;

        void fill_samples() const;

    private:
        entt::entity entity_id;
    };
}
#endif //ENGINE24_KDTREECUDA_H
