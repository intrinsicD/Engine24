

#include "GuiModuleMeshLaplacian.h"
#include "imgui.h"
#include "implot/implot.h"
#include "SurfaceMeshLaplacianOperator.h"
#include "Picker.h"
#include "ModuleMesh.h"
#include "PropertyEigenMap.h"

#include <entt/entity/registry.hpp>

namespace Bcg {
    GuiModuleMeshLaplacian::GuiModuleMeshLaplacian(entt::registry &registry) : GuiModule("Laplacian"),
                                                                               m_registry(registry) {

    }


    void GuiModuleMeshLaplacian::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            if (ImGui::BeginMenu("Mesh")) {
                ImGui::MenuItem(name.c_str(), nullptr, &m_is_window_open);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void GuiModuleMeshLaplacian::render_gui() {
        if (!m_is_window_open) {
            return;
        }


        auto &picker = m_registry.ctx().get<Picked>();
        auto entity_id = picker.entity.id;
        if (entity_id == entt::null && m_registry.valid(entity_id)) {
            m_is_window_open = false;
            return;
        }
        if (!(m_registry.all_of<SurfaceMesh>(entity_id) || m_registry.all_of<MeshHandle>(entity_id))) {
            return;
        }

        auto *mesh = m_registry.try_get<SurfaceMesh>(entity_id);
        if(!mesh){
            auto h_mesh =  m_registry.try_get<MeshHandle>(entity_id);
            if(h_mesh->is_valid()){
                mesh = h_mesh->ptr();
            }
        }

        if (ImGui::Begin("Mesh - Laplacian", &m_is_window_open)) {
            if (ImGui::Button("Compute Cotan Laplacian")) {

                if(mesh){
                    auto laplacians = ComputeCotanLaplacianOperator(*mesh);
                    m_registry.emplace_or_replace<LaplacianMatrices>(entity_id, laplacians);
                }
            }
            if (ImGui::Button("Compute Graph Laplacian")) {
                auto *mesh = m_registry.try_get<SurfaceMesh>(entity_id);
                if(!mesh){
                    auto h_mesh =  m_registry.try_get<MeshHandle>(entity_id);
                    if(h_mesh->is_valid()){
                        mesh = h_mesh->ptr();
                    }
                }
                if(mesh){
                    auto laplacians = ComputeGraphLaplacianOperator(*mesh);
                    m_registry.emplace_or_replace<LaplacianMatrices>(entity_id, laplacians);
                }
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
                        Property<float> evec = mesh->vprops_.add<float>("evec" + std::to_string(i));
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

        }
        ImGui::End();
    }

    // Recursively draws an entity and all its children as a tree.
    void GuiModuleMeshLaplacian::render_gui(entt::entity entity_id) {

    }
}