#include "GuiModuleGaussianMixture.h"

#include "imgui.h"
#include "implot/implot.h"
#include "PointCloudInterface.h"
#include "GraphInterface.h"
#include "Picker.h"
#include "PropertyEigenMap.h"

#include <entt/entity/registry.hpp>

#include "PointCloud.h"

namespace Bcg {
    GuiModuleGaussianMixture::GuiModuleGaussianMixture(entt::registry &registry) : GuiModule("GuiModuleGaussianMixture"),
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
            auto weights = pci->get_vertex_property<float>("v:weights");
            auto scales = pci->get_vertex_property<float>("v:scale");
            auto rotations = pci->get_vertex_property<Vector<float, 4> >("v:rotation");

            if (!covs) {
                if (ImGui::Button("Initialize Covariances")) {
                    if (scales) {
                        if (rotations) {
                            //anisotropic case
                            covs = pci->vertex_property<Matrix<float, 3, 3> >("v:covs");
                            compute_covs_from(covs, scales);
                        }else {
                            //isotropic case
                            covs = pci->vertex_property<Matrix<float, 3, 3> >("v:covs");
                            compute_covs_from(covs, scales);
                        }
                    }
                }
            }else {

            }

        }
        ImGui::End();
    }
}
