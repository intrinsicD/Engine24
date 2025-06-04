//
// Created by alex on 26.11.24.
//

#include "ModuleMesh.h"
#include "ModuleCamera.h"
#include "ModuleTransform.h"
#include "ModuleAABB.h"
#include "ModuleMeshView.h"
#include "ModuleSphereView.h"

#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "PropertiesGui.h"

#include "Engine.h"
#include "ResourcesMesh.h"
#include "SurfaceMeshIo.h" //TODO move to MeshResources.h
#include "Picker.h"
#include "StringUtils.h"
#include "SurfaceMeshCompute.h"

namespace Bcg {
    ModuleMesh::ModuleMesh() : Module("MeshModule") {
    }

    void ModuleMesh::activate() {
        if (base_activate()) {
            if (!Engine::Context().find<MeshPool>()) {
                Engine::Context().emplace<MeshPool>();
            }
            Engine::Dispatcher().sink<Events::Callback::Drop>().connect<&ModuleMesh::on_drop_file>(this);
        }
        ResourcesMesh::activate();
    }

    void ModuleMesh::deactivate() {
        if (base_deactivate()) {
            if (Engine::Context().find<MeshPool>()) {
                Engine::Context().erase<MeshPool>();
            }
            Engine::Dispatcher().sink<Events::Callback::Drop>().disconnect<&ModuleMesh::on_drop_file>(this);
        }
    }

    MeshHandle ModuleMesh::make_handle(const SurfaceMesh &object) {
        auto &pool = Engine::Context().get<MeshPool>();
        return pool.create(object);
    }

    static std::string s_name = "MeshModule";

    MeshHandle ModuleMesh::create(entt::entity entity_id, const SurfaceMesh &object) {
        auto handle = make_handle(object);
        return add(entity_id, handle);
    }

    MeshHandle ModuleMesh::add(entt::entity entity_id, const MeshHandle h_mesh) {
        return Engine::State().get_or_emplace<MeshHandle>(entity_id, h_mesh);
    }

    void ModuleMesh::remove(entt::entity entity_id) {
        Engine::State().remove<MeshHandle>(entity_id);
    }

    bool ModuleMesh::has(entt::entity entity_id) {
        return Engine::State().all_of<MeshHandle>(entity_id);
    }

    MeshHandle ModuleMesh::get(entt::entity entity_id) {
        return Engine::State().get<MeshHandle>(entity_id);
    }

    SurfaceMesh ModuleMesh::load_mesh(const std::string &filepath) {
        return ResourcesMesh::load(filepath);
    }

    bool ModuleMesh::save_mesh(const std::string &filepath, const SurfaceMesh &mesh) {
        if (!Write(filepath, mesh)) {
            std::string ext = filepath;
            ext = ext.substr(ext.find_last_of('.') + 1);
            Log::Error("ModuleMesh: Unsupported file format: " + ext);
            return false;
        }
        return true;
    }

    template<>
    struct GetterTraits<AABB<float>, SurfaceMesh> {
        static AABB<float> getter(const SurfaceMesh &m) noexcept {
            return AABB<float>::Build(m.positions().begin(), m.positions().end());
        }
    };

    void ModuleMesh::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("{}::Setup failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        if (!Engine::State().all_of<MeshHandle>(entity_id)) {
            Log::Warn("{}::Setup failed, entity {} has no MeshHandle.", s_name, static_cast<int>(entity_id));
            return;
        }

        auto h_mesh = get(entity_id);
        auto h_aabb = ModuleAABB::create(entity_id, GetterTraits<AABB<float>, SurfaceMesh>::getter(*h_mesh));
        auto h_transform = ModuleTransform::create(entity_id, Transform::Identity());

        ModuleAABB::center_and_scale_by_aabb(entity_id, h_mesh->vpoint_.name());
        ModuleCamera::center_camera_at_distance(h_aabb->center(), 1.5f * glm::compMax(h_aabb->diagonal()));

        ComputeSurfaceMeshVertexNormals(entity_id);
        //TODO add MeshView etc.
        ModuleMeshView::setup(entity_id);
        ModuleSphereView::setup(entity_id);
        Log::Info("#v: {}, #e: {}, #h: {}, #f: {}",
                  h_mesh->n_vertices(), h_mesh->n_edges(), h_mesh->n_halfedges(), h_mesh->n_faces());
    }

    void ModuleMesh::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("{}::Cleanup failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        if (!Engine::State().all_of<MeshHandle>(entity_id)) {
            Log::Warn("{}::Cleanup failed, Entity {} does not have a MeshHandle. Abort Command", s_name,
                      static_cast<int>(entity_id));
            return;
        }

        remove(entity_id);
    }

    static bool gui_enabled = false;
    static bool file_dialog_gui_enabled = false;

    void ModuleMesh::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            if (ImGui::BeginMenu("Mesh")) {
                if (ImGui::MenuItem("Load Mesh", nullptr, &file_dialog_gui_enabled)) {
                    IGFD::FileDialogConfig config;
                    config.path = ".";
                    config.path = "/home/alex/Dropbox/Work/Datasets";
                    ImGuiFileDialog::Instance()->OpenDialog("Load Mesh", "Choose File", ".obj,.off,.stl,.ply",
                                                            config);
                }
                ImGui::MenuItem("Info", nullptr, &gui_enabled);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    static void render_filedialog() {
        if (ImGuiFileDialog::Instance()->Display("Load Mesh", {}, ImVec2(1050, 750))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto path = ImGuiFileDialog::Instance()->GetFilePathName();
                auto smesh = ModuleMesh::load_mesh(path);
                if (!smesh.is_empty()) {
                    auto entity_id = Engine::State().create();
                    ModuleMesh::create(entity_id, smesh);
                    ModuleMesh::setup(entity_id);
                } else {
                    Log::Error("MeshModule: Failed to load mesh from file: {}", path);
                }
                file_dialog_gui_enabled = false;
            }
            ImGuiFileDialog::Instance()->Close();
        }
    }

    void ModuleMesh::render_gui() {
        if(file_dialog_gui_enabled){
            render_filedialog();
        }
        if (gui_enabled) {
            if (ImGui::Begin("Mesh Info", &gui_enabled, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                show_gui(picked.entity.id);
            }
            ImGui::End();
        }
    }

    void ModuleMesh::show_gui(const MeshHandle &h_mesh) {
        if (h_mesh.is_valid()) {
            show_gui(*h_mesh);
        }
    }

    void ModuleMesh::show_gui(const SurfaceMesh &mesh) {
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

    void ModuleMesh::show_gui(entt::entity entity_id) {
        if (has(entity_id)) {
            show_gui(get(entity_id));
        }
    }

    void ModuleMesh::on_drop_file(const Events::Callback::Drop &event) {
        for (int i = 0; i < event.count; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            SurfaceMesh smesh = load_mesh(event.paths[i]);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> build_duration = end_time - start_time;

            if (smesh.is_empty()) {
                Log::Error("Failed to load mesh from file: {}", event.paths[i]);
                continue;
            }

            auto entity_id = Engine::State().create();
            auto h_mesh = ModuleMesh::create(entity_id, smesh);
            ModuleMesh::setup(entity_id);
            Log::Info("Build Smesh in " + std::to_string(build_duration.count()) + " seconds");
        }
    }
}
