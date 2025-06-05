//
// Created by alex on 15.07.24.
//

#include "ModuleTransform.h"
#include "Engine.h"
#include "Entity.h"
#include "PluginHierarchy.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "Picker.h"
#include "PluginGraphics.h"
#include "CameraUtils.h"


namespace Bcg {

    ModuleTransform::ModuleTransform() : Module("ModuleTransform") {}

    void ModuleTransform::activate() {
        if (base_activate()) {
            if (!Engine::Context().find<TransformPool>()) {
                Engine::Context().emplace<TransformPool>();
            }
        }
    }

    void ModuleTransform::begin_frame() {

    }

    void ModuleTransform::update() {

    }

    void ModuleTransform::end_frame() {

    }

    void ModuleTransform::deactivate() {
        if (base_deactivate()) {
            if (Engine::Context().find<TransformPool>()) {
                Engine::Context().erase<TransformPool>();
            }
        }
    }

    TransformHandle ModuleTransform::make_handle(const Transform &object){
        auto &pool = Engine::Context().get<TransformPool>();
        return pool.create(object);
    }

    TransformHandle ModuleTransform::create(entt::entity entity_id, const Transform &object){
        auto handle = make_handle(object);
        return add(entity_id, handle);
    }

    TransformHandle ModuleTransform::add(entt::entity entity_id, TransformHandle h_object){
        return Engine::State().get_or_emplace<TransformHandle>(entity_id, h_object);
    }

    void ModuleTransform::remove(entt::entity entity_id){
        Engine::State().remove<TransformHandle>(entity_id);
    }

    bool ModuleTransform::has(entt::entity entity_id){
        return Engine::State().all_of<TransformHandle>(entity_id);
    }

    TransformHandle ModuleTransform::get(entt::entity entity_id){
        return Engine::State().get<TransformHandle>(entity_id);
    }

    void ModuleTransform::render() {

    }

    static bool gui_enabled = false;

    void ModuleTransform::render_menu() {
        if (ImGui::BeginMenu("Module")) {
            ImGui::MenuItem("Transform", nullptr, &gui_enabled);
            ImGui::EndMenu();
        }
    }

    void ModuleTransform::render_gui() {
        if (gui_enabled) {
            if (ImGui::Begin(name.c_str(), &gui_enabled, ImGuiWindowFlags_AlwaysAutoResize)) {
                auto &picked = Engine::Context().get<Picked>();
                if (picked.entity) {
                    show_gui(picked.entity.id);
                } else {
                    ImGui::Text("No entity selected");
                }
                ImGui::End();
            }
        }
    }

    bool ModuleTransform::show_gui(TransformHandle &h_transform){
        if (h_transform.is_valid()) {
            return show_gui(*h_transform);
        }
        return false;
    }

    template<typename T>
    inline void ShowMatrix(const Matrix<T, 4, 4> &m) {
        ImGui::Text("%f %f %f %f\n"
                    "%f %f %f %f\n"
                    "%f %f %f %f\n"
                    "%f %f %f %f",
                    m[0][0], m[1][0], m[2][0], m[3][0],
                    m[0][1], m[1][1], m[2][1], m[3][1],
                    m[0][2], m[1][2], m[2][2], m[3][2],
                    m[0][3], m[1][3], m[2][3], m[3][3]);
    }

    bool Equals(const glm::mat4 &m1, const glm::mat4 &m2) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (m1[i][j] != m2[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool ApplyGizmo(glm::mat4 &objectMatrix, bool &is_scaling) {
        // Copy the original matrix (so Manipulate can write over `m`).
        glm::mat4 m = objectMatrix;

        // MODE (Local vs World) and OPERATION (Translate/Rotate/Scale)
        static ImGuizmo::MODE currentGizmoMode      = ImGuizmo::LOCAL;
        static ImGuizmo::OPERATION currentGizmoOp   = ImGuizmo::ROTATE;

        // Radio buttons for operation
        if (ImGui::RadioButton("Translate", currentGizmoOp == ImGuizmo::TRANSLATE))
            currentGizmoOp = ImGuizmo::TRANSLATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate",    currentGizmoOp == ImGuizmo::ROTATE))
            currentGizmoOp = ImGuizmo::ROTATE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale",     currentGizmoOp == ImGuizmo::SCALE))
            currentGizmoOp = ImGuizmo::SCALE;
        ImGui::SameLine();
        if (ImGui::RadioButton("Universal", currentGizmoOp == ImGuizmo::UNIVERSAL))
            currentGizmoOp = ImGuizmo::UNIVERSAL;

        ImGui::Separator();

        // Decide whether scaling UI is allowed
        is_scaling = (currentGizmoOp == ImGuizmo::SCALE);

        // MODE (Local vs World) only if not scaling; if scaling you might lock orientation
        if (!is_scaling) {
            if (ImGui::RadioButton("Local", currentGizmoMode == ImGuizmo::LOCAL)) {
                currentGizmoMode = ImGuizmo::LOCAL;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("World", currentGizmoMode == ImGuizmo::WORLD)) {
                currentGizmoMode = ImGuizmo::WORLD;
            }
        }

        // Snapping
        static bool use_snap = false;
        static float snap[3] = {1.0f, 1.0f, 1.0f};
        ImGui::Checkbox("Use Snap", &use_snap);
        ImGui::SameLine();
        if (currentGizmoOp == ImGuizmo::TRANSLATE) {
            ImGui::InputFloat3("Snap", &snap[0]);
        } else if (currentGizmoOp == ImGuizmo::ROTATE) {
            ImGui::InputFloat("Angle Snap", &snap[0]);
        } else if (currentGizmoOp == ImGuizmo::SCALE) {
            ImGui::InputFloat("Scale Snap", &snap[0]);
        } else {
            ImGui::Text("Snap not supported for this mode");
        }

        // Optional bounding box
        static bool use_bounds           = false;
        static bool use_bounds_snapting  = false;
        static float bounds[6]           = {-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f};
        static float bounds_snap[3]      = {0.1f, 0.1f, 0.1f};
        ImGui::Checkbox("Use Bound Sizing", &use_bounds);
        ImGui::Checkbox("Use Bound Snap",   &use_bounds_snapting);

        // Get camera matrices
        auto &camera = Engine::Context().get<Camera>();
        const float* viewPtr = glm::value_ptr(camera.view);
        const float* projPtr = glm::value_ptr(camera.proj);

        // Set the area where the gizmo lives (must match your 3D viewport)
        auto   win_pos   = PluginGraphics::get_window_pos();
        auto   win_size  = PluginGraphics::get_window_size();
        ImGuizmo::SetRect(win_pos.x, win_pos.y, win_size.x, win_size.y);

        // Perform the manipulation: ImGuizmo writes into `m` if user drags
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::Manipulate(
                viewPtr,
                projPtr,
                currentGizmoOp,
                currentGizmoMode,
                glm::value_ptr(m),
                nullptr,                           // we do not need the delta‐matrix directly
                use_snap        ? &snap[0]       : nullptr,
                use_bounds      ? bounds         : nullptr,
                use_bounds_snapting ? bounds_snap : nullptr
        );

        // If the gizmo is actively being used, ImGuizmo::IsUsing() = true
        if (ImGuizmo::IsUsing()) {
            objectMatrix = m;  // write back the new transform
            return true;
        }
        return false;
    }

    bool Show(TransformParameters &t_params) {
        ImGui::PushID(&t_params);
        ImGui::Text("Scale");
        bool changed = ImGui::InputFloat3("##Scale", glm::value_ptr(t_params.scale));
        ImGui::Text("Rotation");
        changed |= ImGui::InputFloat3("##Rotation", glm::value_ptr(t_params.angle_axis));
        ImGui::Text("Translation");
        changed |= ImGui::InputFloat3("##Translation", glm::value_ptr(t_params.position));
        ImGui::PopID();
        return changed;
    }

    static bool show_guizmo = false;

    bool ModuleTransform::show_gui(Transform &transform){
        bool changed = false;
        ImGui::Text("Local");
        ShowMatrix(transform.local());
        ImGui::Separator();
        TransformParameters t_params = decompose(transform.local());
        if (Show(t_params)) {
            transform.set_local(compose(t_params));
            changed = true;
        }
        ImGui::Text("World");
        ShowMatrix(transform.world());
        ImGui::Separator();
        TransformParameters w_params = decompose(transform.world());
        Show(w_params);
        if (ImGui::CollapsingHeader("Cached Parent World")) {
            ShowMatrix(transform.get_cached_parent_world());
        }
        // Checkbox to toggle the gizmo
        ImGui::Checkbox("Show Guizmo", &show_guizmo);
        if (show_guizmo) {
            bool is_scaling = false;
            glm::mat4 local_mat = transform.local();

            // Apply the ImGuizmo handle on local_mat
            if (ApplyGizmo(local_mat, is_scaling)) {
                transform.set_local(local_mat);
                changed = true;
            }
        }
        return changed;
    }

    bool ModuleTransform::show_gui(entt::entity entity_id){
        if (Engine::valid(entity_id) && Engine::has<TransformHandle>(entity_id)) {
            return show_gui(Engine::State().get<TransformHandle>(entity_id));
        } else {
            ImGui::Text("Entity %d has no Transform component", static_cast<int>(entity_id));
        }
        return false;
    }

    Transform *ModuleTransform::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return nullptr; }
        if (Engine::has<Transform>(entity_id)) { return &Engine::State().get<Transform>(entity_id); }

        Log::Info("Transform setup for entity: {}", entity_id);
        return &Engine::State().emplace<Transform>(entity_id, Transform());
    }

    void ModuleTransform::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return; }
        if (!Engine::has<Transform>(entity_id)) { return; }

        Engine::State().remove<Transform>(entity_id);
        Log::Info("Transform cleanup for entity: {}", entity_id);
    }

    void ModuleTransform::set_identity_transform(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return; }
        if (!Engine::has<Transform>(entity_id)) { return; }
        Engine::State().get<Transform>(entity_id).set_local(glm::mat4(1.0f));

        PluginHierarchy::mark_transforms_dirty(entity_id);
    }
}