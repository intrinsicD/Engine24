#include "GuiModuleGaussianMixture.h"

#include "imgui.h"
#include "implot/implot.h"
#include "PointCloudInterface.h"
#include "PointCloud.h"
#include "GraphInterface.h"
#include "GmmUtils.h"
#include "Picker.h"
#include "Engine.h"
#include "AABBsToGraph.h"
#include "GeometryUtils.h"
#include "PropertyEigenMap.h"
#include "TransformComponent.h"
#include "ModuleGraph.h"
#include "PointCloudUtils.h"
#include "GaussianGalerkinLaplacian.h"
#include "VecTraits.h"

#include <entt/entity/registry.hpp>

#include "EventsKeys.h"


namespace Bcg {
    GuiModuleGaussianMixture::GuiModuleGaussianMixture(
        entt::registry &registry) : GuiModule("GuiModuleGaussianMixture"),
                                    m_registry(registry) {
    }


    void GuiModuleGaussianMixture::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            if (ImGui::BeginMenu("Graph")) {
                ImGui::MenuItem(name.c_str(), nullptr, &m_is_window_open);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void GuiModuleGaussianMixture::render_gui() {
        if (!m_is_window_open) {
            return;
        }

        auto &picker = m_registry.ctx().get<Picked>();
        auto entity_id = picker.entity.id;
        render_gui(entity_id);
    }

    // Recursively draws an entity and all its children as a tree.
    void GuiModuleGaussianMixture::render_gui(entt::entity entity_id) {
        if (entity_id == entt::null && m_registry.valid(entity_id)) {
            m_is_window_open = false;
            return;
        }
        if (!(m_registry.all_of<PointCloud>(entity_id) || m_registry.all_of<PointCloudInterface>(entity_id))) {
            return;
        }

        auto *pci = m_registry.try_get<PointCloudInterface>(entity_id);
        if (!pci) {
            auto *pc = m_registry.try_get<PointCloud>(entity_id);
            if (pc) {
                pci = &pc->interface;
            }
        }

        if (!pci) {
            m_is_window_open = false;
            return;
        }

        if (ImGui::Begin("Gaussian Mixture", &m_is_window_open)) {
            auto mus = pci->vertex_property<PointType>("v:point");
            auto covs = pci->get_vertex_property<Matrix<float, 3, 3> >("v:covs");
            // Fix: retrieve the correct inverse covariance property
            auto covs_inv = pci->get_vertex_property<Matrix<float, 3, 3> >("v:covs_inv");
            auto weights = pci->get_vertex_property<float>("v:weights");
            auto scales = pci->get_vertex_property<Vector<float, 3> >("v:scale");
            auto rotations = pci->get_vertex_property<Vector<float, 4> >("v:rotation");

            static float sigma_k = 1.0f;
            if (scales) {
                ImGui::SliderFloat("Sigma multiplier (k)", &sigma_k, 0.1f, 5.0f, "%.2f x sigma");
                ImGui::SameLine();
                if (ImGui::SmallButton("1x")) sigma_k = 1.0f;
                ImGui::SameLine();
                if (ImGui::SmallButton("2x")) sigma_k = 2.0f;
                ImGui::SameLine();
                if (ImGui::SmallButton("3x")) sigma_k = 3.0f;
            }

            static std::vector<Gaussian<float>> gaussians(mus.vector().size());
            static LaplacianAssemblyOptions<float> opt;
            static float c_value = 1.0f;
            ImGui::InputFloat("c value", &c_value);
            ImGui::Checkbox("Use mass lumping", &opt.lump_mass);
            ImGui::Checkbox("Use random-walk normalization", &opt.normalize_rw);
            ImGui::Checkbox("Use symmetric normalization", &opt.normalize_sym);
            ImGui::Checkbox("Use symmetric pairs", &opt.symmetric_pair);
            if (ImGui::Button("Galerkin")) {
                for (size_t i = 0; i < mus.vector().size(); ++i) {
                    gaussians[i].mu = Eigen::Vector<float, 3>(mus.vector()[i].x, mus.vector()[i].y, mus.vector()[i].z);
                    gaussians[i].log_sigma = Eigen::Vector<float, 3>(scales.vector()[i].x, scales.vector()[i].y, scales.vector()[i].z);
                    gaussians[i].q = Eigen::Quaternion<float>(rotations.vector()[i].x, rotations.vector()[i].y, rotations.vector()[i].z, rotations.vector()[i].w).normalized();
                    gaussians[i].w = weights ? weights.vector()[i] : 1.0f;
                }

                auto result = AssembleGaussianGalerkinLaplacian<float>(gaussians, opt, c_value, sigma_k);
                Engine::State().emplace_or_replace<LaplacianMatrices>(entity_id, result);
            }

            if(m_registry.all_of<LaplacianMatrices>(entity_id)){
                auto &laplacians = m_registry.get<LaplacianMatrices>(entity_id);
                static int k = 10;
                ImGui::InputInt("num eigenvalues", &k);
                static bool generalized = true;
                ImGui::Checkbox("generalized", &generalized);
                static float sigma = 0.0f;
                ImGui::InputFloat("sigma", &sigma);
                if(ImGui::Button("Compute Eigen Decomposition")){
                    auto result = EigenDecompositionSparse(laplacians, k, generalized, sigma);
                    m_registry.emplace_or_replace<EigenDecompositionResult>(entity_id, result);
                    for(int i = 0; i < k; ++i){
                        Property<float> evec = pci->vertices.add<float>("evec" + std::to_string(i));
                        Map(evec.vector()) = result.evecs.col(i);
                    }
                }
                if(m_registry.all_of<EigenDecompositionResult>(entity_id)){
                    auto &result = m_registry.get<EigenDecompositionResult>(entity_id);
                    //Show plot of eigenvalues if they exist
                    if (ImPlot::BeginPlot("Eigenvalue Spectrum", "Index (i)", "Value (λ)")) {
                        ImPlot::PlotLine("λ", result.evals.data(), result.evals.size());
                        ImPlot::EndPlot();
                    }

                    // --- Method 2: Log-Scale Plot ---
                    if (ImPlot::BeginPlot("Log-Scale Spectrum", "Index (i)", "Value (log λ)")) {
                        // --- FIX ---
                        // 1. Create a temporary vector for the shifted data.
                        std::vector<float> shifted_evals;
                        shifted_evals.resize(result.evals.size());

                        // 2. Add a small epsilon to each eigenvalue.
                        float epsilon = 1e-8f;
                        for (size_t i = 0; i < result.evals.size(); ++i) {
                            shifted_evals[i] = result.evals[i] + epsilon;
                        }

                        // 3. Set the axis scale to logarithmic *before* plotting.
                        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);

                        // 4. Plot the new, shifted data.
                        ImPlot::PlotLine("log(λ + ε)", shifted_evals.data(), shifted_evals.size());

                        ImPlot::EndPlot();
                    }
                }
                ImGui::Separator();
            }

            if (covs) {
                if (ImGui::Button("Visualize Covariances as boxes")) {
                    auto aabbs_id = Engine::State().create();
                    auto &gi = Require<GraphInterface>(aabbs_id, Engine::State());
                    gi.clear();
                    if (scales && rotations) {
                        // Visualize oriented k-sigma boxes (k defaults to 1; set to 3 for 3-sigma)
                        OrientedBoxesToGraph(mus.vector(), scales.vector(), rotations.vector(), gi, sigma_k);
                    } else if (scales) {
                        // Fallback: no rotations, visualize axis-aligned boxes scaled by k-sigma
                        std::vector<AABB<float> > aabbs;
                        aabbs.reserve(mus.vector().size());
                        const float k = std::max(0.0f, sigma_k);
                        for (size_t i = 0; i < mus.vector().size(); ++i) {
                            const auto &mu = mus.vector()[i];
                            const auto &s = scales.vector()[i];
                            Vector<float, 3> ext(k * s.x, k * s.y, k * s.z);
                            AABB<float> box;
                            box.min = mu - ext;
                            box.max = mu + ext;
                            aabbs.emplace_back(box);
                        }
                        AABBsToGraph(aabbs, gi);
                    }
                    ModuleGraph::setup(aabbs_id);
                    auto transform_entity = Engine::State().get<TransformComponent>(entity_id);
                    auto &transform_boxes = Engine::require<TransformComponent>(aabbs_id);
                    transform_boxes = transform_entity;
                }
                if (ImGui::Button("Visualize Covariances as local frames")) {
                    compute_vectorfields_of_gaussians(*pci);
                }
            }
            if (!covs_inv && covs) {
                if (ImGui::Button("Compute Inverse Covariances")) {
                    covs_inv = pci->vertex_property<Matrix<float, 3, 3> >("v:covs_inv");
                    covs_inv.vector() = compute_covs_inverse_from(covs.vector());
                }
                //TODO continue here...
            }
        }
        ImGui::End();
    }
}
