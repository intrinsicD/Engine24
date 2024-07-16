//
// Created by alex on 08.07.24.
//

#include "PluginCamera.h"
#include "imgui.h"
#include "Engine.h"
#include "Keyboard.h"
#include "EventsCallbacks.h"
#include "EventsKeys.h"
#include "Timer.h"
#include "Mouse.h"
#include "Picker.h"
#include "Graphics.h"
#include "PluginFrameTimer.h"
#include "../MatVec.h"
#include "Logger.h"
#include "AABB.h"
#include "Eigen/Geometry"
#include "glad/gl.h"
#include "CameraGui.h"

namespace Bcg {

    PluginCamera::PluginCamera() : Plugin("Camera") {}

    void PluginCamera::transform(Camera &camera, const Matrix<float, 4, 4> &transformation) {
        auto &v_params = camera.v_params;
        v_params.eye = (transformation * v_params.eye.homogeneous()).head<3>();
        v_params.center = (transformation * v_params.center.homogeneous()).head<3>();
        v_params.up = transformation.block<3, 3>(0, 0) * v_params.up;
    }

    static Vector<int, 2> last_point_2d_;
    static Vector<float, 3> last_point_3d_;
    static bool last_point_ok_ = false;

    static bool map_to_sphere(const Vector<int, 2> &point2D, Vector<float, 3> &result) {
        auto vp = Graphics::get_viewport();
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

    static void rotate(Camera &camera, const Vector<float, 3> &axis, float angle) {
        // center in eye coordinates

        Vector<float, 4> ec = camera.view * camera.v_params.center.homogeneous();
        Vector<float, 3> c(ec[0] / ec[3], ec[1] / ec[3], ec[2] / ec[3]);
        Matrix<float, 4, 4> center_matrix = translation_matrix(c);
        Matrix<float, 4, 4> rot_matrix = rotation_matrix(axis, angle);

        camera.view = center_matrix * rot_matrix * center_matrix.inverse() * camera.view;
        Matrix<float, 4, 4> inv_view = camera.view.inverse();
        camera.dirty_view = true;
        camera.v_params.eye = inv_view.block<3, 1>(0, 3);
        camera.v_params.up = inv_view.block<3, 1>(0, 1);
    }

    static void rotation(Camera &camera, int x, int y) {
        if (last_point_ok_) {
            Vector<int, 2> newPoint2D;
            Vector<float, 3> newPoint3D;
            bool newPointok;

            newPoint2D = Vector<int, 2>(x, y);
            newPointok = map_to_sphere(newPoint2D, newPoint3D);

            if (newPointok) {
                Vector<float, 3> axis = cross(last_point_3d_, newPoint3D);
                float cosAngle = dot(last_point_3d_, newPoint3D);

                if (fabs(cosAngle) < 1.0) {
                    float angle = 2.0 * acos(cosAngle) * 180.0 / std::numbers::pi;
                    rotate(camera, axis, angle);
                }
            }
        }
    }

/*    static void translate(Camera &camera, const Vector<float, 3> &t) {
        camera.view = translation_matrix(t) * camera.view;
        Matrix<float, 4, 4> inv_view = camera.view.inverse();
        camera.dirty_view = true;

        camera.v_params.eye = inv_view.block<3, 1>(0, 3);
        camera.v_params.center -= t;
    }*/

    static void translate(Camera &camera, const Vector<float, 3> &t) {
        camera.v_params.eye = (translation_matrix(t) * camera.v_params.eye.homogeneous()).head<3>();
        camera.v_params.center = (translation_matrix(t) * camera.v_params.center.homogeneous()).head<3>();
        camera.v_params.dirty = true;
    }

    static void translate(Camera &camera, int x, int y) {
        float dx = x - last_point_2d_[0];
        float dy = y - last_point_2d_[1];

        auto vp = Graphics::get_viewport();
        double w = vp[2];
        double h = vp[3];

        Vector<float, 4> ec = camera.view * camera.v_params.center.homogeneous();
        float z = (ec[2] / ec[3]);

        float aspect = (float) w / (float) h;
        float up = tan(camera.p_params.fovy / 2.0f * std::numbers::pi / 180.f) * camera.p_params.zNear;
        float right = aspect * up;

        translate(camera, Vector<float, 3>(2.0 * dx / w * right / camera.p_params.zNear * z,
                                           -2.0 * dy / h * up / camera.p_params.zNear * z, 0.0f));
    }

    static void on_mouse_cursor(const Events::Callback::MouseCursor &event) {
        auto &mouse = Engine::Context().get<Mouse>();
        auto &camera = Engine::Context().get<Camera>();
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
        auto &camera = Engine::Context().get<Camera>();
        const float min_fovy = 1.0f;   // Minimum field of view in degrees
        const float max_fovy = 45.0f;  // Maximum field of view in degrees

        // Adjust the field of view based on scroll input
        camera.p_params.fovy -= event.yoffset;

        // Ensure the field of view stays within reasonable bounds
        camera.p_params.fovy = std::clamp(camera.p_params.fovy, min_fovy, max_fovy);
        camera.p_params.dirty = true;
    }

    static void on_key_focus(const Events::Key::F &event) {
        if (event.action) {
            auto &picked = Engine::Context().get<Picked>();
            auto &camera = Engine::Context().get<Camera>();
            auto diff = camera.v_params.eye - camera.v_params.center;

            camera.v_params.center = picked.world_space_point;
            camera.v_params.eye = camera.v_params.center + diff;
            camera.v_params.dirty = true;
            Log::Info("Focus onto: (" + std::to_string(camera.v_params.center[0]) + ", " +
                      std::to_string(camera.v_params.center[1]) + ", " + std::to_string(camera.v_params.center[2]) +
                      ")");
        }
    }

    static void on_key_center(const Events::Key::C &event) {
        if (event.action) {
            auto &picked = Engine::Context().get<Picked>();
            auto &camera = Engine::Context().get<Camera>();
            if (Engine::valid(picked.entity.id)) {
                auto &aabb = Engine::State().get<AABB<float>>(picked.entity.id);
                float d = aabb.diagonal().maxCoeff() / tan(camera.p_params.fovy / 2.0);
                camera.v_params.center = aabb.center();
                camera.v_params.eye = camera.v_params.center + Vector<float, 3>(0.0, 0.0, d);
                camera.v_params.up = camera.v_params.center + Vector<float, 3>(0.0, 1.0, 0);
                camera.v_params.dirty = true;
                Log::Info("Center onto: (" + std::to_string(camera.v_params.center[0]) + ", " +
                          std::to_string(camera.v_params.center[1]) + ", " + std::to_string(camera.v_params.center[2]) +
                          ")");
            }
        }
    }

    static void on_window_resize(const Events::Callback::WindowResize &event) {
        auto &camera = Engine::Context().get<Camera>();
        if (camera.proj_type == Camera::ProjectionType::PERSPECTIVE) {
            camera.p_params.aspect = float(event.width) / float(event.height);
            camera.p_params.dirty = true;
        } else {
            float half_width = event.width * 0.5f;
            float half_height = event.height * 0.5f;
            camera.o_params.left = -half_width;
            camera.o_params.right = half_width;
            camera.o_params.top = half_height;
            camera.o_params.bottom = -half_height;
            camera.o_params.dirty = true;
        }
    }

    void PluginCamera::activate() {
        if (!Engine::Context().find<Camera>()) {
            auto &camera = Engine::Context().emplace<Camera>();
            auto vp = Graphics::get_viewport();
            camera.p_params.aspect = float(vp[2]) / float(vp[3]);
            camera.p_params.dirty = true;
        }
        if (!Engine::Context().find<CameraUniformBuffer>()) {
            auto &ubo = Engine::Context().emplace<CameraUniformBuffer>();
            glGenBuffers(1, &ubo.id);
            glBindBuffer(GL_UNIFORM_BUFFER, ubo.id);
            glBufferData(GL_UNIFORM_BUFFER, 2 * sizeof(Matrix<float, 4, 4>), NULL, GL_STATIC_DRAW);
            glBindBuffer(GL_UNIFORM_BUFFER, 0);
            // Bind the UBO to the binding point
            glBindBufferBase(GL_UNIFORM_BUFFER, ubo.binding_point, ubo.id);
        }
        Engine::Dispatcher().sink<Events::Callback::MouseCursor>().connect<&on_mouse_cursor>();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().connect<&on_mouse_scroll>();
        Engine::Dispatcher().sink<Events::Callback::WindowResize>().connect<&on_window_resize>();
        Engine::Dispatcher().sink<Events::Key::F>().connect<&on_key_focus>();
        Engine::Dispatcher().sink<Events::Key::C>().connect<&on_key_center>();
        Plugin::activate();
    }

    void PluginCamera::begin_frame() {
        auto &camera = Engine::Context().get<Camera>();
        auto &keyboard = Engine::Context().get<Keyboard>();
        auto dt = PluginFrameTimer::delta();
        Vector<float, 3> front = (camera.v_params.center - camera.v_params.eye).normalized();
        if (keyboard.w()) {
            translate(camera, front * dt);
            camera.v_params.dirty = true;
        }
        if (keyboard.s()) {
            translate(camera, -front * dt);
            camera.v_params.dirty = true;
        }
        Vector<float, 3> right = cross(front, camera.v_params.up).normalized();
        if (keyboard.a()) {
            translate(camera, -right * dt);
            camera.v_params.dirty = true;
        }
        if (keyboard.d()) {
            translate(camera, right * dt);
            camera.v_params.dirty = true;
        }
    }

    void PluginCamera::update() {
        auto &camera = Engine::Context().get<Camera>();
        auto &ubo = Engine::Context().get<CameraUniformBuffer>();

        if (camera.v_params.dirty) {
            camera.view = look_at_matrix(camera.v_params.eye, camera.v_params.center, camera.v_params.up);
            camera.v_params.dirty = false;
            camera.dirty_view = true;
        }
        if (camera.proj_type == Camera::ProjectionType::PERSPECTIVE) {
            if (camera.p_params.dirty) {
                camera.proj = perspective_matrix(camera.p_params.fovy, camera.p_params.aspect, camera.p_params.zNear,
                                                 camera.p_params.zFar);
                camera.p_params.dirty = false;
                camera.dirty_proj = true;
            }
        } else {
            if (camera.o_params.dirty) {
                camera.proj = ortho_matrix(camera.o_params.left, camera.o_params.right, camera.o_params.bottom,
                                           camera.o_params.top, camera.o_params.zNear, camera.o_params.zFar);
                camera.o_params.dirty = false;
                camera.dirty_proj = true;
            }
        }

        glBindBuffer(GL_UNIFORM_BUFFER, ubo.id);
        if (camera.dirty_view) {
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(camera.view), camera.view.data());
            camera.dirty_view = false;
        }

        if (camera.dirty_proj) {
            glBufferSubData(GL_UNIFORM_BUFFER, sizeof(camera.view), sizeof(camera.proj), camera.proj.data());
            camera.dirty_proj = false;
        }
        glBindBuffer(GL_UNIFORM_BUFFER, 0);
    }

    void PluginCamera::end_frame() {

    }

    void PluginCamera::deactivate() {
        if (Engine::Context().find<CameraUniformBuffer>()) {
            auto &ubo = Engine::Context().get<CameraUniformBuffer>();
            glDeleteBuffers(1, &ubo.id);
        }
        Engine::Context().erase<CameraUniformBuffer>();
        Engine::Dispatcher().sink<Events::Callback::MouseCursor>().disconnect<&on_mouse_cursor>();
        Engine::Dispatcher().sink<Events::Callback::MouseScroll>().disconnect<&on_mouse_scroll>();
        Engine::Dispatcher().sink<Events::Callback::WindowResize>().disconnect<&on_window_resize>();
        Engine::Dispatcher().sink<Events::Key::F>().disconnect<&on_key_focus>();
        Engine::Dispatcher().sink<Events::Key::C>().disconnect<&on_key_center>();
        Plugin::deactivate();
    }

    static bool show_gui = false;

    void PluginCamera::render_menu() {
        if (ImGui::BeginMenu("Menu")) {
            ImGui::MenuItem(name, nullptr, &show_gui);
            ImGui::EndMenu();
        }
    }

    void PluginCamera::render_gui() {
        if (show_gui) {
            if (ImGui::Begin(name, &show_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                Gui::Show(Engine::Context().get<Camera>());
                ImGui::End();
            }
        }
    }

    void PluginCamera::render() {

    }
}