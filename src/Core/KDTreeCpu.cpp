//
// Created by alex on 01.08.24.
//

#include "Types.h"
#include "KDTreeCpu.h"

namespace Bcg {
    KDTreeCpu::KDTreeCpu() : index(nullptr) {

    }

    KDTreeCpu::~KDTreeCpu() {

    }

    void KDTreeCpu::build(const std::vector<Vector<float, 3>> &positions) {
        dataset = std::make_unique<VectorAdapter>(positions);
        index = std::make_unique<Type>(3, *dataset,
                                       nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        index->buildIndex();
    }

    QueryResult KDTreeCpu::knn_query(const Vector<float, 3> &query_point, unsigned int num_closest) const {
        int k = num_closest + 1;
        QueryResult result(k);
        nanoflann::KNNResultSet<float> resultSet(k);
        resultSet.init(result.indices.data(), result.distances.data());
        nanoflann::SearchParameters params;
        params.sorted = true;
        params.eps = 0.0;
        index->findNeighbors(resultSet, &query_point[0], params);
        result.indices = std::vector(result.indices.begin() + 1, result.indices.end());
        result.distances = std::vector(result.distances.begin() + 1, result.distances.end());
        return result;
    }

    QueryResult KDTreeCpu::radius_query(const Vector<float, 3> &query_point, float radius) const {
        nanoflann::SearchParameters params;
        std::vector<nanoflann::ResultItem<size_t, float>> items;
        nanoflann::RadiusResultSet resultSet(radius * radius, items);
        resultSet.init();

        index->findNeighbors(resultSet, &query_point[0], params);

        size_t nMatches = items.size();
        QueryResult result(nMatches);
        for (size_t i = 0; i < nMatches; ++i) {
            result.indices[i] = items[i].first;
            result.distances[i] = items[i].second;
        }

        return result;
    }

    QueryResult KDTreeCpu::closest_query(const Vector<float, 3> &query_point) const {
        return knn_query(query_point, 1);
    }

    float KDTreeCpu::VectorAdapter::kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx][dim];
    }

    std::vector<QueryResult> KDTreeCpu::knn_query_batch(const std::vector<Vector<float, 3>> &query_points, unsigned int num_closest) const{
        std::vector<QueryResult> results(query_points.size());
        for (size_t i = 0; i < query_points.size(); ++i) {
            results[i] = knn_query(query_points[i], num_closest);
        }
        return results;
    }

    std::vector<QueryResult> KDTreeCpu::radius_query_batch(const std::vector<Vector<float, 3>> &query_points, float radius) const{
        std::vector<QueryResult> results(query_points.size());
        for (size_t i = 0; i < query_points.size(); ++i) {
            results[i] = radius_query(query_points[i], radius);
        }
        return results;
    }

    std::vector<QueryResult> KDTreeCpu::closest_query_batch(const std::vector<Vector<float, 3>> &query_points) const{
        std::vector<QueryResult> results(query_points.size());
        for (size_t i = 0; i < query_points.size(); ++i) {
            results[i] = closest_query(query_points[i]);
        }
        return results;
    }
}