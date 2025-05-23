//
// Created by alex on 08.07.24.
//

#include "PluginCamera.h"
#include "imgui.h"
#include "Engine.h"
#include "Entity.h"
#include "Keyboard.h"
#include "EventsCallbacks.h"
#include "EventsKeys.h"
#include "Mouse.h"
#include "Keyboard.h"
#include "Picker.h"
#include "PluginGraphics.h"
#include "PluginFrameTimer.h"
#include "Logger.h"
#include "AABB.h"
#include "BoundingVolumes.h"
#include "Eigen/Geometry"
#include "CameraGui.h"
#include "Transform.h"

namespace Bcg {
    //TODO fix camera, and setup aspect on camera creation etc...

    PluginCamera::PluginCamera() : Plugin("Camera") {}

    Camera<float> *PluginCamera::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return nullptr; }
        if (Engine::has<Camera<float>>(entity_id)) { return &Engine::State().get<Camera<float>>(entity_id); }

        Log::Info("Camera setup for entity: {}", entity_id);
        auto &camera = Engine::State().emplace<Camera<float>>(entity_id, Camera<float>());
        setup(camera);

        return &camera;
    }

    void PluginCamera::setup(Camera<float> &camera) {
        auto view_params = camera.get_view_params();
        view_params.eye = Eigen::Vector<float, 3>(0, 0, 1);
        view_params.center = Eigen::Vector<float, 3>(0, 0, 0);
        view_params.up = Eigen::Vector<float, 3>(0, 1, 0);
        camera.set_view_params(view_params);

        auto vp = PluginGraphics::get_viewport();
        auto viewport_width = vp[2];
        auto viewport_height = vp[3];

        float aspect_ratio = float(viewport_width) / float(viewport_height);
        auto p_params = camera.get_perspective_params();
        p_params.fovy_degrees = 45.0f;
        p_params.aspect = aspect_ratio;
        p_params.zNear = 0.1f;
        p_params.zFar = 100.0f;
        camera.set_perspective_params(p_params);
    }

    void PluginCamera::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return; }
        if (!Engine::has<Camera<float>>(entity_id)) { return; }

        Engine::State().remove<Camera<float>>(entity_id);
    }

    static Eigen::Vector<int, 2> last_point_2d_;
    static Eigen::Vector<float, 3> last_point_3d_;
    static bool last_point_ok_ = false;

    static bool map_to_sphere(const Eigen::Vector<int, 2> &point2D, Eigen::Vector<float, 3> &result) {
        auto vp = PluginGraphics::get_viewport();
        double w = vp[2];
        double h = vp[3];
        if ((point2D[0] >= 0) && (point2D[0] <= w) && (point2D[1] >= 0) &&
            (point2D[1] <= h)) {
            double x = (double) (point2D[0] - 0.5 * w) / w;
            double y = (double) (0.5 * h - point2D[1]) / h;
            double sinx = sin(std::numbers::pi * x * 0.5);
            double siny = sin(std::numbers::pi * y * 0.5);
            double sinx2siny2 = sinx * sinx + siny * siny;

            result[0] = sinx;
            result[1] = siny;
            result[2] = sinx2siny2 < 1.0 ? sqrt(1.0 - sinx2siny2) : 0.0;

            return true;
        } else
            return false;
    }

    static void rotate(Camera<float> &camera, const Eigen::Vector<float, 3> &axis, float angle) {
        // center in eye coordinates

        auto view_params = camera.get_view_params();
        Eigen::Vector<float, 4> ec = camera.get_view() * Eigen::Vector<float, 4>(view_params.center, 1.0f);
        Eigen::Vector<float, 3> c(ec[0] / ec[3], ec[1] / ec[3], ec[2] / ec[3]);

        Eigen::Matrix<float, 4, 4> view = Eigen::Translation<float, 3>(c) * Eigen::AngleAxis<float>(angle,axis) * Eigen::Translation<float, 3>(-c) * camera.get_view();
        Eigen::Matrix<float, 4, 4> model_matrix = view.inverse();
        view_params.eye = Eigen::Vector<float, 3>(model_matrix.col(3));
        view_params.up = Eigen::Vector<float, 3>(model_matrix.col(1));
        camera.set_view_params(view_params);
    }

    static void rotation(Camera<float> &camera, int x, int y) {
        if (last_point_ok_) {
            Eigen::Vector<int, 2> newPoint2D;
            Eigen::Vector<float, 3> newPoint3D;
            bool newPointok;

            newPoint2D = Eigen::Vector<int, 2>(x, y);
            newPointok = map_to_sphere(newPoint2D, newPoint3D);

            if (newPointok) {
                Eigen::Vector<float, 3> axis = last_point_3d_.cross(newPoint3D);
                float cosAngle = last_point_3d_.dot(newPoint3D);

                if (fabs(cosAngle) < 1.0) {
                    float angle = 2.0 * acos(cosAngle) * 180.0 / std::numbers::pi;
                    rotate(camera, axis, angle);
                }
            }
        }
    }

    static void translate(Camera<float> &camera, const Eigen::Vector<float, 3> &t) {
        auto view_params = camera.get_view_params();
        view_params.eye += t;
        view_params.center += t;
        camera.set_view_params(view_params);
    }

    static void translate(Camera<float> &camera, int x, int y) {
        float dx = x - last_point_2d_[0];
        float dy = y - last_point_2d_[1];

        //translate the camera in worldspace in the image plane
        auto view_params = camera.get_view_params();
        Eigen::Vector<float, 3> front = view_params.center - view_params.eye;
        Eigen::Vector<float, 3> up = view_params.up;
        Eigen::Vector<float, 3> right = front.cross(up);

        float distance_to_scene = front.norm();
        // Project the change in screen coordinates to world coordinates
        auto p_params = camera.get_perspective_params();
        auto vp = PluginGraphics::get_viewport();
        float viewport_width = vp[2];
        float viewport_height = vp[3];
        float fov = p_params.fovy_degrees; // Field of view in radians

        // Compute the scale factors for screen to world space translation
        float aspect_ratio = viewport_width / viewport_height;
        float half_fov_y = fov / 2.0;
        float half_fov_x = std::atan(std::tan(half_fov_y) * aspect_ratio);

        float world_dx = dx / viewport_width * 2 * std::tan(half_fov_x) * distance_to_scene;
        float world_dy = dy / viewport_height * 2 * std::tan(half_fov_y) * distance_to_scene;

        // Translate the camera in world space
        Eigen::Vector<float, 3> translation = up * world_dy - right * world_dx;
        view_params.center += translation;
        view_params.eye += translation;

        // Update the last_point_2d_ to the current cursor position
        last_point_2d_[0] = x;
        last_point_2d_[1] = y;
        camera.set_view_params(view_params);
    }

    static void on_mouse_cursor(const Events::Callback::MouseCursor &event) {
        auto &mouse = Engine::Context().get<Mouse>();
        auto &camera = Engine::Context().get<Camera<float>>();
        if (mouse.left()) {
            //rotate camera around center like an arc ball camera
            rotation(camera, event.xpos, event.ypos);
        }
        if (mouse.middle()) {
            translate(camera, event.xpos, event.ypos);
        }

        // remember points
        last_point_2d_ = {event.xpos, event.ypos};
        last_point_ok_ = map_to_sphere(last_point_2d_, last_point_3d_);
    }

    static void on_mouse_scroll(const Events::Callback::MouseScroll &event) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (keyboard.strg()) return;

        auto &camera = Engine::Context().get<Camera<float>>();
        const float min_fovy = 1.0f;   // Minimum field of view in degrees
        const float max_fovy = 45.0f;  // Maximum field of view in degrees

        // Adjust the field of view based on scroll input
        auto p_params = camera.get_perspective_params();
        p_params.fovy_degrees -= event.yoffset;

        // Ensure the field of view stays within reasonable bounds
        p_params.fovy_degrees = std::clamp(p_params.fovy_degrees, min_fovy, max_fovy);
        camera.set_perspective_params(p_params);
    }

    static void on_key_focus(const Events::Key::F &event) {
        if (event.action) {
            auto &picked = Engine::Context().get<Picked>();
            auto &camera = Engine::Context().get<Camera<float>>();
            auto view_params = camera.get_view_params();
            Eigen::Vector<float, 3> front = view_params.center - view_params.eye;

            float d = front.norm();
            view_params.center = picked.spaces.wsp;
            view_params.eye = view_params.center - front * d;
            camera.set_view_params(view_params);
            Log::Info("Focus onto: {} {} {}", view_params.center[0], view_params.center[1], view_params.center[2]);
        }
    }

    static void on_key_center(const Events::Key::C &event) {
        if (event.action) {
            auto &picked = Engine::Context().get<Picked>();
            auto &camera = Engine::Context().get<Camera<float>>();
            if (Engine::valid(picked.entity.id)) {
                auto &bv = Engine::State().get<BoundingVolumes>(picked.entity.id);
                auto &aabb = *bv.h_aabb;
                auto p_params = camera.get_perspective_params();
                auto view_params = camera.get_view_params();
                float d = aabb.diagonal().maxCoeff() /*/ tan(p_params.fovy / 2.0)*/;
                Eigen::Vector<float, 3> front = view_params.center - view_params.eye;
                view_params.center = aabb.center();
                view_params.eye = view_params.center - front * d;
                camera.set_view_params(view_params);
                Log::Info("Center onto: {} {} {}", view_params.center[0], view_params.center[1], view_params.center[2]);
            }
        }
    }

    static void on_framebuffer_resize(const Events::Callback::FramebufferResize &event) {
        auto &camera = Engine::Context().get<Camera<float>>();
        if (camera.get_projection_type() == Camera<float>::ProjectionType::ORTHOGRAPHIC) {
            float half_width = event.width * 0.5f;
            float half_height = event.height * 0.5f;
            auto o_params = camera.get_ortho_params();
            o_params.left = -half_width;
            o_params.right = half_width;
            o_params.top = half_height;
            o_params.bottom = -half_height;
            camera.set_ortho_params(o_params);
        } else {
            auto p_params = camera.get_perspective_params();
            p_params.aspect = float(event.width) / float(event.height);
            camera.set_perspective_params(p_params);
        }
    }

    void PluginCamera::activate() {
        auto &camera = Engine::Context().emplace<Camera<float>>();
        setup(camera);

        if (!Engine::Context().find<CameraUniformBuffer>()) {
            auto &ubo = Engine::Context().emplace<CameraUniformBuffer>();
            ubo.create();
            ubo.bind();
            ubo.buffer_data(nullptr, 2 * sizeof(Eigen::Matrix<float, 4, 4>), Bcg::Buffer::Usage::STATIC_DRAW);
            ubo.unbind();
            ubo.bind_base(0);
        }
        Engine::Dispatcher().sink<Events::Callback::MouseCursor>().connect<&on_mouse_cursor>();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().connect<&on_mouse_scroll>();
        Engine::Dispatcher().sink<Events::Callback::FramebufferResize>().connect<&on_framebuffer_resize>();
        Engine::Dispatcher().sink<Events::Key::F>().connect<&on_key_focus>();
        Engine::Dispatcher().sink<Events::Key::C>().connect<&on_key_center>();
        Plugin::activate();
    }

    void PluginCamera::begin_frame() {
        auto &camera = Engine::Context().get<Camera<float>>();
        auto &keyboard = Engine::Context().get<Keyboard>();
        auto dt = PluginFrameTimer::delta();
        auto view_params = camera.get_view_params();
        Eigen::Vector<float, 3> front = view_params.center - view_params.eye;
        if (keyboard.w()) {
            translate(camera, front * dt);
        }
        if (keyboard.s()) {
            translate(camera, -front * dt);
        }
        Eigen::Vector<float, 3> right = front.cross(view_params.up);
        if (keyboard.a()) {
            translate(camera, -right * dt);
        }
        if (keyboard.d()) {
            translate(camera, right * dt);
        }
    }

    void PluginCamera::update() {
        auto &camera = Engine::Context().get<Camera<float>>();
        auto &ubo = Engine::Context().get<CameraUniformBuffer>();
        auto view_params = camera.get_view_params();

        ubo.bind();
        ubo.buffer_sub_data(camera.get_view().data(), sizeof(camera.get_view()));
        ubo.buffer_sub_data(camera.get_proj().data(), sizeof(camera.get_proj()), sizeof(camera.get_view()));
        ubo.unbind();
    }

    void PluginCamera::end_frame() {

    }

    void PluginCamera::deactivate() {
        if (Engine::Context().find<CameraUniformBuffer>()) {
            auto &ubo = Engine::Context().get<CameraUniformBuffer>();
            ubo.destroy();
        }
        Engine::Context().erase<CameraUniformBuffer>();
        Engine::Dispatcher().sink<Events::Callback::MouseCursor>().disconnect<&on_mouse_cursor>();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().disconnect<&on_mouse_scroll>();
        Engine::Dispatcher().sink<Events::Callback::FramebufferResize>().disconnect<&on_framebuffer_resize>();
        Engine::Dispatcher().sink<Events::Key::F>().disconnect<&on_key_focus>();
        Engine::Dispatcher().sink<Events::Key::C>().disconnect<&on_key_center>();
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginCamera::render_menu() {
        if (ImGui::BeginMenu("Graphics")) {
            ImGui::MenuItem(name, nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginCamera::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name, &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::Show(Engine::Context().get<Camera<float>>());
                ImGui::End();
            }
        }
    }

    void PluginCamera::render() {

    }

    namespace Commands {
        void CenterCameraAtDistance::execute() const {
            auto &camera = Engine::Context().get<Camera<float>>();
            auto view_params = camera.get_view_params();
            auto p_params = camera.get_perspective_params();

            Eigen::Vector<float, 3> front = view_params.center - view_params.eye;
            view_params.center = center;
            view_params.eye = center - front * distance / tanf(p_params.fovy_degrees / 2.0f);
            camera.set_view_params(view_params);
            FitNearAndFarToDistance(distance).execute();
        }

        void FitNearAndFarToDistance::execute() const {
            auto &camera = Engine::Context().get<Camera<float>>();

            if (camera.get_projection_type() == Camera<float>::ProjectionType::PERSPECTIVE) {
                auto p_params = camera.get_perspective_params();
                p_params.zNear = distance / 100.0f;
                p_params.zFar = distance * 3.0f;
                camera.set_perspective_params(p_params);
            } else if (camera.get_projection_type() == Camera<float>::ProjectionType::ORTHOGRAPHIC) {
                auto o_params = camera.get_ortho_params();
                o_params.zNear = distance / 100.0f;
                o_params.zFar = distance * 3.0f;
                camera.set_ortho_params(o_params);
            }
        }
    }
}