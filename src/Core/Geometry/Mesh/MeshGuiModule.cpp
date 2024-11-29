//
// Created by alex on 26.11.24.
//

#include "MeshGuiModule.h"
#include "imgui.h"
#include "Logger.h"
#include "Engine.h"
#include "Pool.h"
#include "MeshStringTraits.h"
#include "PropertiesGui.h"
#include "MeshModule.h"
#include "ImGuiFileDialog.h"
#include "Picker.h"

namespace Bcg {
    MeshGuiModule::MeshGuiModule() : GuiModule("MeshGuiModule") {}

    void MeshGuiModule::activate() {
        if(base_activate()){
            Engine::Dispatcher().sink<Events::Gui::Menu::Render>().connect<&MeshGuiModule::on_render_menu>(this);
            Engine::Dispatcher().sink<Events::Gui::Render>().connect<&MeshGuiModule::on_render>(this);
        }
    }

    void MeshGuiModule::deactivate() {
        if(base_deactivate()){
            Engine::Dispatcher().sink<Events::Gui::Menu::Render>().disconnect<&MeshGuiModule::on_render_menu>(this);
            Engine::Dispatcher().sink<Events::Gui::Render>().disconnect<&MeshGuiModule::on_render>(this);
        }
    }

    void MeshGuiModule::render_filedialog() {
        if (ImGuiFileDialog::Instance()->Display("Load Mesh", ImGuiWindowFlags_NoCollapse, ImVec2(200, 100))) {
            if (ImGuiFileDialog::Instance()->IsOk()) { // action if OK
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
                // action
                MeshModule meshModule;
                auto mesh = meshModule.load_mesh(filePathName);
                meshModule.make_handle(mesh);

            }

            // close
            ImGuiFileDialog::Instance()->Close();
        }
    }

    void MeshGuiModule::render(const MeshComponent &meshes) {
        render(meshes.current_mesh);
        for (const auto &h_mesh: meshes.meshes) {
            if (ImGui::TreeNodeEx(std::to_string(h_mesh.get_index()).c_str())) {
                render(h_mesh);
                ImGui::TreePop();
            }
        }
    }

    void MeshGuiModule::render(const PoolHandle<SurfaceMesh> &handle) {
        if (ImGui::CollapsingHeader(
                (std::string("Show Properties ###") + std::to_string(handle.get_index())).c_str())) {
            auto &pool = Engine::Context().get<Pool<SurfaceMesh> >();
            render(*handle);
        }
        render(*handle);
    }

    void MeshGuiModule::render(const SurfaceMesh &mesh) {
        if (ImGui::CollapsingHeader(("Vertices #v: " + std::to_string(mesh.n_vertices())).c_str())) {
            ImGui::PushID("Vertices");
            Gui::Show("##Vertices", mesh.vprops_);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader(("Halfedges #h: " + std::to_string(mesh.n_halfedges())).c_str())) {
            ImGui::PushID("Halfedges");
            Gui::Show("##Halfedges", mesh.hprops_);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader(("Edges #e: " + std::to_string(mesh.n_edges())).c_str())) {
            ImGui::PushID("Edges");
            Gui::Show("##Edges", mesh.eprops_);
            ImGui::PopID();
        }
        if (ImGui::CollapsingHeader(("Faces #f: " + std::to_string(mesh.n_faces())).c_str())) {
            ImGui::PushID("Faces");
            Gui::Show("##Faces", mesh.fprops_);
            ImGui::PopID();
        }
    }

    void MeshGuiModule::render(Pool<SurfaceMesh> &pool) {
        Gui::Show("MeshPoolProperties", pool.properties);
    }

    void MeshGuiModule::render(entt::entity entity_id) {
        if (Engine::has<MeshComponent>(entity_id)) {
            auto &component = Engine::State().get<MeshComponent>(entity_id);
            render(component);
        }
    }

    static bool show_mesh_instance_gui = false;
    static bool show_mesh_pool_gui = false;

    void MeshGuiModule::render_menu() {
        if (ImGui::BeginMenu("Entity Module")) {
            if (ImGui::BeginMenu("Mesh")) {
                if (ImGui::MenuItem("Load Mesh")) {
                    IGFD::FileDialogConfig config;
                    config.path = ".";
                    config.path = "/home/alex/Dropbox/Work/Datasets";
                    ImGuiFileDialog::Instance()->OpenDialog("Load Mesh", "Choose File", ".obj,.off,.stl,.ply",
                                                            config);
                }
                if (ImGui::MenuItem("Instance", nullptr, &show_mesh_instance_gui)) {

                }
                if (ImGui::MenuItem("Pool", nullptr, &show_mesh_pool_gui)) {

                }
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    void MeshGuiModule::render_gui() {
        render_filedialog();
        if (show_mesh_instance_gui) {
            auto &picked = Engine::Context().get<Picked>();
            if (ImGui::Begin("Mesh", &show_mesh_instance_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                render(picked.entity.id);
            }
            ImGui::End();
        }
        if(show_mesh_pool_gui){
            if (ImGui::Begin("Pool", &show_mesh_pool_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &pool = Engine::Context().get<Pool<SurfaceMesh> >();
                render(pool);
            }
            ImGui::End();
        }
    };

    void MeshGuiModule::register_events(entt::dispatcher &dispatcher){
        Engine::Dispatcher().sink<Events::Gui::Menu::Render>().connect<&MeshGuiModule::on_render_menu>(this);
        Engine::Dispatcher().sink<Events::Gui::Render>().connect<&MeshGuiModule::on_render>(this);
        Log::Info("MeshGuiModule registered events");
    }

    void MeshGuiModule::unregister_events(entt::dispatcher &dispatcher){
        Engine::Dispatcher().sink<Events::Gui::Menu::Render>().disconnect<&MeshGuiModule::on_render_menu>(this);
        Engine::Dispatcher().sink<Events::Gui::Render>().disconnect<&MeshGuiModule::on_render>(this);
        Log::Info("MeshGuiModule unregistered events");
    }

    void MeshGuiModule::on_render_menu(const Events::Gui::Menu::Render &event){

    }

    void MeshGuiModule::on_render(const Events::Gui::Render &event){

    }
}