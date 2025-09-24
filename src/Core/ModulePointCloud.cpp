//
// Created by alex on 12.08.25.
//

#include "ModulePointCloud.h"

#include "ModuleCamera.h"
#include "TransformComponent.h"
#include "TransformUtils.h"
#include "ModuleAABB.h"
#include "ModuleSphereView.h"
#include "ModulePhongSplattingView.h"

#include "imgui.h"
#include "ImGuiFileDialog.h"
#include "PropertiesGui.h"

#include "StringTraitsPointCloud.h"
#include "Engine.h"
#include "ResourcesPointCloud.h"
#include "PointCloudIo.h"
#include "PointCloudToGraph.h"
#include "CommandsPointCloud.h"
#include "Picker.h"
#include "GeometryUtils.h"

#include "ModuleGraphView.h"

namespace Bcg {
    ModulePointCloud::ModulePointCloud() : Module("PointCloudModule") {
    }

    void ModulePointCloud::activate() {
        if (base_activate()) {
            Engine::Dispatcher().sink<Events::Callback::Drop>().connect<&ModulePointCloud::on_drop_file>(this);
        }
        ResourcesPointCloud::activate();
    }

    void ModulePointCloud::deactivate() {
        if (base_deactivate()) {
            Engine::Dispatcher().sink<Events::Callback::Drop>().disconnect<&ModulePointCloud::on_drop_file>(this);
        }
    }

    static std::string s_name = "PointCloudModule";

    void ModulePointCloud::remove(entt::entity entity_id) {
        Engine::State().remove<PointCloudInterface>(entity_id);
    }

    bool ModulePointCloud::has(entt::entity entity_id) {
        return Engine::State().all_of<PointCloudInterface>(entity_id);
    }

    void ModulePointCloud::destroy_entity(entt::entity entity_id) {
        remove(entity_id);
        if (Engine::State().all_of<Vertices>(entity_id)) {
            Engine::State().remove<Vertices>(entity_id);
        }
    }

    PointCloud ModulePointCloud::load_point_cloud(const std::string &filepath) {
        return ResourcesPointCloud::load(filepath);
    }

    bool ModulePointCloud::save_point_cloud(const std::string &filepath, const Bcg::PointCloud &pc) {
        if (!Write(filepath, pc)) {
            std::string ext = filepath;
            ext = ext.substr(ext.find_last_of('.') + 1);
            Log::Error("ModulePointCloud: Unsupported file format: " + ext);
            return false;
        }
        return true;
    }

    template<>
    struct BuilderTraits<AABB<float>, PointCloudInterface> {
        static AABB<float> build(const PointCloudInterface &pci) noexcept {
            return AABB<float>::Build(pci.vpoint.vector().begin(), pci.vpoint.vector().end());
        }
    };

    void ModulePointCloud::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("{}::Setup failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        auto &pci = Require<PointCloudInterface>(entity_id, Engine::State());

        auto h_aabb = ModuleAABB::create(entity_id, BuilderTraits<AABB<float>, PointCloudInterface>::build(pci));

        auto &transform = Engine::require<TransformComponent>(entity_id);

        //ScaleAndCenterAt(transform, h_aabb->center(), 1 / glm::compMax(h_aabb->diagonal()));

        //ModuleAABB::center_and_scale_by_aabb(entity_id, pci.vpoint.name());
        ModuleCamera::center_camera_at_distance(h_aabb->center(),  glm::compMax(h_aabb->diagonal()));

        Commands::ComputePointCloudLocalPcasKnn(entity_id, 32).execute();
        //TODO add ComputeSurfacePointCloudVertexNormals etc.
        ModuleSphereView::setup(entity_id);
        ModulePhongSplattingView::setup(entity_id);
        Log::Info("#v: {}", pci.vertices.n_vertices());
    }

    void ModulePointCloud::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) {
            Log::Warn("{}::Cleanup failed, Entity is not valid. Abort Command", s_name);
            return;
        }

        if (!Engine::State().all_of<PointCloudInterface>(entity_id)) {
            Log::Warn("{}::Cleanup failed, Entity {} does not have a PointCloudHandle. Abort Command", s_name,
                      static_cast<int>(entity_id));
            return;
        }

        remove(entity_id);
    }

    static bool gui_enabled = false;
    static bool file_dialog_gui_enabled = false;
    static bool gui_to_graph = false;
    static bool gui_graph_laplacian = false;

    void ModulePointCloud::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            if (ImGui::BeginMenu("PointCloud")) {
                if (ImGui::MenuItem("Load PointCloud", nullptr, &file_dialog_gui_enabled)) {
                    IGFD::FileDialogConfig config;
                    config.path = ".";
                    config.path = "/home/alex/Dropbox/Work/Datasets";
                    ImGuiFileDialog::Instance()->OpenDialog("Load PointCloud", "Choose File", ".csv, .pts, .xyz",
                                                            config);
                }
                ImGui::MenuItem("Info", nullptr, &gui_enabled);
                ImGui::MenuItem("To Graph", nullptr, &gui_to_graph);
                ImGui::EndMenu();
            }
            ImGui::EndMenu();
        }
    }

    static void render_filedialog() {
        if (ImGuiFileDialog::Instance()->Display("Load PointCloud", {}, ImVec2(1050, 750))) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                auto path = ImGuiFileDialog::Instance()->GetFilePathName();
                auto spc = ModulePointCloud::load_point_cloud(path);
                if (!spc.interface.is_empty()) {
                    auto entity_id = Engine::State().create();
                    auto &vertices = Require<Vertices>(entity_id, Engine::State());
                    vertices = spc.data.vertices;
                    auto &pci = Require<PointCloudInterface>(entity_id, Engine::State());
                    ModulePointCloud::setup(entity_id);
                } else {
                    Log::Error("PointCloudModule: Failed to load pc from file: {}", path);
                }
                file_dialog_gui_enabled = false;
            }
            ImGuiFileDialog::Instance()->Close();
        }
    }

    void ModulePointCloud::render_gui() {
        if(file_dialog_gui_enabled){
            render_filedialog();
        }
        if (gui_enabled) {
            if (ImGui::Begin("PointCloud Info", &gui_enabled, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                show_gui(picked.entity.id);
            }
            ImGui::End();
        }
        if (gui_to_graph) {
            if (ImGui::Begin("PointCloud To Graph", &gui_to_graph, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                const auto entity_id = picked.entity.id;
                if (has(entity_id)) {
                    auto &pci = Require<PointCloudInterface>(entity_id, Engine::State());
                    if (ImGui::CollapsingHeader("Knn - Graph")) {
                        static int num_closest = 6;
                        ImGui::InputInt("num_closest", &num_closest);
                        if (ImGui::Button("Create##KnnGraph")) {
                            auto &gi = Require<GraphInterface>(entity_id, Engine::State());
                            PointCloudToKNNGraph(pci, num_closest, gi);
                            ModuleGraphView::setup(entity_id);
                        }
                        ImGui::Separator();
                    }
                    if (ImGui::CollapsingHeader("Radius - Graph")) {
                        static float radius = 0.1f;
                        ImGui::InputFloat("radius", &radius);
                        if (ImGui::Button("Create##RadiusGraph")) {
                            auto &gi = Require<GraphInterface>(entity_id, Engine::State());
                            PointCloudToRadiusGraph(pci, radius, gi);
                            ModuleGraphView::setup(entity_id);
                        }
                    }


                }
            }
            ImGui::End();
        }
    }

    void ModulePointCloud::show_gui(const PointCloudInterface &pci) {
        if (ImGui::CollapsingHeader(("Vertices #v: " + std::to_string(pci.vertices.n_vertices())).c_str())) {
            ImGui::PushID("Vertices");
            Gui::Show("##Vertices", pci.vertices);
            ImGui::PopID();
        }
    }

    void ModulePointCloud::show_gui(entt::entity entity_id) {
        if (has(entity_id)) {
            show_gui(Require<PointCloudInterface>(entity_id, Engine::State()));
        }
    }

    void ModulePointCloud::on_drop_file(const Events::Callback::Drop &event) {
        for (int i = 0; i < event.count; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            PointCloud spc = load_point_cloud(event.paths[i]);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> build_duration = end_time - start_time;

            if (spc.interface.is_empty()) {
                Log::Error("Failed to load pc from file: {}", event.paths[i]);
                continue;
            }

            auto entity_id = Engine::State().create();
            auto &vertices = Require<Vertices>(entity_id, Engine::State());
            vertices = spc.data.vertices;
            auto &pci = Require<PointCloudInterface>(entity_id, Engine::State());
            setup(entity_id);
            Log::Info("Build Spc in " + std::to_string(build_duration.count()) + " seconds");
        }
    }
}
