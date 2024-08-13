//
// Created by alex on 13.08.24.
//

#ifndef ENGINE24_ICP_H
#define ENGINE24_ICP_H

#include <string>
#include "MatVec.h"
#include "entt/fwd.hpp"
#include "Types.h"

namespace Bcg {
    class SamplingStrategyBase {
    public:
        SamplingStrategyBase(std::string name, entt::entity entity_id) : name(std::move(name)), entity_id(entity_id) {}

        virtual ~SamplingStrategyBase() = default;

        const std::vector<Vector<float, 3>> get_samples() const;

        std::string name;
        entt::entity entity_id;
    };

    class CorresponodenceStrategyBase {
    public:
        CorresponodenceStrategyBase(std::string name, entt::entity entity_id) : name(std::move(name)), entity_id(entity_id) {}

        virtual ~CorresponodenceStrategyBase() = default;

        const std::vector<std::vector<IndexType, WeightType>> get_correspondences() const;

        std::string name;
        entt::entity entity_id;
    };

    class WeightingStrategyBase {

    };

    class OutlierRejectionStrategyBase {

    };

    class EstimateTransformationStrategyBase {

    };

    struct IcpConfig {
        std::shared_ptr<SamplingStrategyBase> source_sampling_strategy;
        std::shared_ptr<SamplingStrategyBase> target_sampling_strategy;
        std::shared_ptr<CorresponodenceStrategyBase> corresponodence_strategy;
        std::shared_ptr<WeightingStrategyBase> weighting_strategy;
        std::shared_ptr<OutlierRejectionStrategyBase> outlier_rejection_strategy;
        std::shared_ptr<EstimateTransformationStrategyBase> estimate_transformation_strategy;

        int max_iterations = 100;
        float tolerance = 1e-5;
    };


    class Icp {

    };
}

#endif //ENGINE24_ICP_H
