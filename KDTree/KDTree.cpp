//
// Created by alex on 01.08.24.
//

#include "Types.h"
#include "KDtree.h"

namespace Bcg {
    KDTree::KDTree() : index(nullptr) {

    }

    KDTree::~KDTree() {

    }

    void KDTree::build(const std::vector<Vector<float, 3>> &positions) {
        dataset = std::make_unique<VectorAdapter>(positions);
        index = std::make_unique<Type>(3, *dataset,
                                       nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        index->buildIndex();
    }

    QueryResult KDTree::knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const {
        QueryResult result;
        result.indices.resize(num_closest);
        result.distances.resize(num_closest);
        nanoflann::KNNResultSet<float> resultSet(num_closest);
        resultSet.init(result.indices.data(), result.distances.data());
        index->findNeighbors(resultSet, &query_point[0], nanoflann::SearchParameters(10));
        return result;
    }

    QueryResult KDTree::radius_query(const Vector<float, 3> &query_point, float radius) const {
        QueryResult result;
        nanoflann::SearchParameters params;
        std::vector<nanoflann::ResultItem<size_t, float>> items;
        nanoflann::RadiusResultSet resultSet(radius, items);
        resultSet.init();

        index->findNeighbors(resultSet, &query_point[0], params);

        size_t nMatches = items.size();
        result.indices.resize(nMatches);
        result.distances.resize(nMatches);
        for (size_t i = 0; i < nMatches; ++i) {
            result.indices[i] = items[i].first;
            result.distances[i] = items[i].second;
        }

        return result;
    }

    QueryResult KDTree::closest_query(const Vector<float, 3> &query_point) const {
        return knn_query(query_point, 1);
    }

    float KDTree::VectorAdapter::kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx][dim];
    }
}