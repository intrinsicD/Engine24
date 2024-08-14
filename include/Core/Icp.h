//
// Created by alex on 13.08.24.
//

#ifndef ENGINE24_ICP_H
#define ENGINE24_ICP_H

#include <string>
#include "entt/fwd.hpp"
#include "Types.h"
#include "RigidTransform.h"

namespace Bcg {
    class SamplingStrategyBase {
    public:
        SamplingStrategyBase(std::string name, entt::entity entity_id) : name(std::move(name)),
                                                                         entity_id(entity_id) {}

        virtual ~SamplingStrategyBase() = default;

        virtual const std::vector<Vector<float, 3>> get_samples() const;

        std::string name;
        entt::entity entity_id;
    };

    class CorresponodenceStrategyBase {
    public:
        CorresponodenceStrategyBase(std::string name, entt::entity entity_id) : name(std::move(name)),
                                                                                entity_id(entity_id) {}

        virtual ~CorresponodenceStrategyBase() = default;

        virtual const std::vector<std::vector<IndexType, WeightType>> get_correspondences() const;

        std::string name;
        entt::entity entity_id;
    };

    class WeightingStrategyBase {
        WeightingStrategyBase(std::string name, entt::entity entity_id) : name(std::move(name)),
                                                                          entity_id(entity_id) {}

        virtual ~WeightingStrategyBase() = default;

        virtual const std::vector<WeightType> get_weights() const;

        std::string name;
        entt::entity entity_id;
    };

    class OutlierRejectionStrategyBase {
        OutlierRejectionStrategyBase(std::string name, entt::entity entity_id) : name(std::move(name)),
                                                                                 entity_id(entity_id) {}

        virtual ~OutlierRejectionStrategyBase() = default;

        virtual const std::vector<IndexType> get_inliers() const;

        std::string name;
        entt::entity entity_id;
    };

    class EstimateTransformationStrategyBase {
        EstimateTransformationStrategyBase(std::string name, entt::entity entity_id) : name(std::move(name)),
                                                                                       entity_id(entity_id) {}

        virtual ~EstimateTransformationStrategyBase() = default;

        virtual const RigidTransform get_estimated_transform() const;

        std::string name;
        entt::entity entity_id;
    };

    struct IcpConfig {
        entt::entity source, target;
        RigidTransform delta;
        std::vector<float> errors;

        int max_iterations = 100;
        float tolerance = 1e-5;

        std::shared_ptr<SamplingStrategyBase> source_sampling_strategy;
        std::shared_ptr<SamplingStrategyBase> target_sampling_strategy;
        std::shared_ptr<CorresponodenceStrategyBase> corresponodence_strategy;
        std::shared_ptr<WeightingStrategyBase> weighting_strategy;
        std::shared_ptr<OutlierRejectionStrategyBase> outlier_rejection_strategy;
        std::shared_ptr<EstimateTransformationStrategyBase> estimate_transformation_strategy;
    };

    class Icp {
    public:
        IcpConfig icp_config;

        RigidTransform step(){
            auto source_samples = icp_config.source_sampling_strategy->get_samples();
            auto target_samples = icp_config.target_sampling_strategy->get_samples();
            auto correspondences = icp_config.corresponodence_strategy->get_correspondences();
        }

        RigidTransform iterate();
    };
}

#endif //ENGINE24_ICP_H
