//
// Created by alex on 08.07.24.
//

#include "ModuleCamera.h"
#include "CameraUtils.h"
#include "Engine.h"
#include "Entity.h"
#include "Keyboard.h"
#include "EventsCallbacks.h"
#include "EventsKeys.h"
#include "Mouse.h"
#include "Picker.h"
#include "ModuleGraphics.h"
#include "PluginFrameTimer.h"
#include "ModuleAABB.h"

namespace Bcg {
    //TODO fix camera, and setup aspect on camera creation etc...

    ModuleCamera::ModuleCamera() : Module("Camera") {
    }

    Camera *ModuleCamera::setup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return nullptr; }
        if (Engine::has<Camera>(entity_id)) { return &Engine::State().get<Camera>(entity_id); }

        Log::Info("Camera setup for entity: {}", entity_id);
        auto &camera = Engine::State().emplace<Camera>(entity_id, Camera());
        setup(camera);

        return &camera;
    }

    void ModuleCamera::setup(Camera &camera) {
        ViewParams view_params = GetViewParams(camera);
        view_params.eye = glm::vec3(0, 0, 1);
        view_params.center = glm::vec3(0, 0, 0);
        view_params.up = glm::vec3(0, 1, 0);
        camera.proj_type = Camera::ProjectionType::PERSPECTIVE;
        SetViewParams(camera, view_params);

        auto vp = ModuleGraphics::get_viewport();
        auto viewport_width = vp[2];
        auto viewport_height = vp[3];

        float aspect_ratio = float(viewport_width) / float(viewport_height);
        PerspectiveParams p_params = GetPerspectiveParams(camera);
        p_params.fovy_degrees = 45.0f;
        p_params.aspect = aspect_ratio;
        p_params.zNear = 0.1f;
        p_params.zFar = 100.0f;
        SetPerspectiveParams(camera, p_params);
    }

    void ModuleCamera::cleanup(entt::entity entity_id) {
        if (!Engine::valid(entity_id)) { return; }
        if (!Engine::has<Camera>(entity_id)) { return; }

        Engine::State().remove<Camera>(entity_id);
    }

    static Vector<int, 2> last_point_2d_;
    static glm::vec3 last_point_3d_;
    static bool last_point_ok_ = false;

    static bool map_to_sphere(const Vector<int, 2> &point2D, glm::vec3 &result) {
        auto vp = ModuleGraphics::get_viewport();
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

    static void rotate(Camera &camera, const glm::vec3 &axis, float angle) {
        // center in eye coordinates

        ViewParams v_params = GetViewParams(camera);
        glm::vec4 ec = camera.view * glm::vec4(v_params.center, 1.0f);
        glm::vec3 c(ec[0] / ec[3], ec[1] / ec[3], ec[2] / ec[3]);
        glm::mat4 center_matrix = glm::translate(glm::mat4(1.0f), -c);
        glm::mat4 rot_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(angle), axis);

        camera.view = center_matrix * rot_matrix * glm::inverse(center_matrix) * camera.view;
        glm::mat4 inv_view = glm::inverse(camera.view);
        v_params.eye = glm::vec3(inv_view[3]);
        v_params.up = glm::vec3(inv_view[1]);
        SetViewParams(camera, v_params);
    }

    static void rotation(Camera &camera, int x, int y) {
        if (last_point_ok_) {
            Vector<int, 2> newPoint2D;
            glm::vec3 newPoint3D;
            bool newPointok;

            newPoint2D = Vector<int, 2>(x, y);
            newPointok = map_to_sphere(newPoint2D, newPoint3D);

            if (newPointok) {
                glm::vec3 axis = cross(last_point_3d_, newPoint3D);
                float cosAngle = glm::dot(last_point_3d_, newPoint3D);

                if (fabs(cosAngle) < 1.0) {
                    float angle = 2.0 * acos(cosAngle) * 180.0 / std::numbers::pi;
                    rotate(camera, axis, -angle);
                }
            }
        }
    }

    static void translate(Camera &camera, const glm::vec3 &t) {
        ViewParams v_params = GetViewParams(camera);
        v_params.eye += t;
        v_params.center += t;
        SetViewParams(camera, v_params);
    }

    static void translate(Camera &camera, int x, int y) {
        float dx = x - last_point_2d_[0];
        float dy = y - last_point_2d_[1];

        //translate the camera in worldspace in the image plane
        ViewParams v_params = GetViewParams(camera);
        glm::vec3 front = v_params.center - v_params.eye;
        glm::vec3 up = v_params.up;
        glm::vec3 right = glm::cross(front, up);

        float distance_to_scene = glm::length(front);
        // Project the change in screen coordinates to world coordinates
        PerspectiveParams p_params = GetPerspectiveParams(camera);
        auto vp = ModuleGraphics::get_viewport();
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
        glm::vec3 translation = up * world_dy - right * world_dx;
        v_params.center += translation;
        v_params.eye += translation;

        // Update the last_point_2d_ to the current cursor position
        last_point_2d_[0] = x;
        last_point_2d_[1] = y;
        SetViewParams(camera, v_params);
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
        auto &keyboard = Engine::Context().get<Keyboard>();
        if (keyboard.strg()) return;

        auto &camera = Engine::Context().get<Camera>();
        const float min_fovy = 1.0f; // Minimum field of view in degrees
        const float max_fovy = 45.0f; // Maximum field of view in degrees

        // Adjust the field of view based on scroll input
        PerspectiveParams p_params = GetPerspectiveParams(camera);
        p_params.fovy_degrees -= event.yoffset;

        // Ensure the field of view stays within reasonable bounds
        p_params.fovy_degrees = std::clamp(p_params.fovy_degrees, min_fovy, max_fovy);
        SetPerspectiveParams(camera, p_params);
    }

    static void on_key_focus(const Events::Key::F &event) {
        if (event.action) {
            auto &picked = Engine::Context().get<Picked>();
            auto &camera = Engine::Context().get<Camera>();
            ViewParams v_params = GetViewParams(camera);
            glm::vec3 front = v_params.center - v_params.eye;

            float d = glm::length(front);
            v_params.center = picked.spaces.wsp;
            v_params.eye = v_params.center - front * d;
            SetViewParams(camera, v_params);
            Log::Info("Focus onto: {} {} {}", v_params.center[0], v_params.center[1], v_params.center[2]);
        }
    }

    static void on_key_center(const Events::Key::C &event) {
        if (event.action) {
            auto &picked = Engine::Context().get<Picked>();
            auto &camera = Engine::Context().get<Camera>();
            if (Engine::valid(picked.entity.id)) {
                auto h_aabb = ModuleAABB::get(picked.entity.id);
                PerspectiveParams p_params = GetPerspectiveParams(camera);
                ViewParams v_params = GetViewParams(camera);
                float d = glm::compMax(h_aabb->diagonal()) /*/ tan(p_params.fovy / 2.0)*/;
                glm::vec3 front = v_params.center - v_params.eye;
                v_params.center = h_aabb->center();
                v_params.eye = v_params.center - front * d;
                SetViewParams(camera, v_params);
                Log::Info("Center onto: {} {} {}", v_params.center[0], v_params.center[1], v_params.center[2]);
            }
        }
    }

    static void on_framebuffer_resize(const Events::Callback::FramebufferResize &event) {
        auto &camera = Engine::Context().get<Camera>();
        if (camera.proj_type == Camera::ProjectionType::ORTHOGRAPHIC) {
            float half_width = event.width * 0.5f;
            float half_height = event.height * 0.5f;
            OrthoParams o_params = GetOrthoParams(camera);
            o_params.left = -half_width;
            o_params.right = half_width;
            o_params.top = half_height;
            o_params.bottom = -half_height;
            SetOrthoParams(camera, o_params);
        } else {
            PerspectiveParams p_params = GetPerspectiveParams(camera);
            p_params.aspect = float(event.width) / float(event.height);
            SetPerspectiveParams(camera, p_params);
        }
    }

    void ModuleCamera::activate() {
        if (base_activate()) {
            auto &camera = Engine::Context().emplace<Camera>();
            setup(camera);

            if (!Engine::Context().find<CameraUniformBuffer>()) {
                auto &ubo = Engine::Context().emplace<CameraUniformBuffer>();
                ubo.create();
                ubo.bind();
                ubo.buffer_data(nullptr, 2 * sizeof(glm::mat4), Bcg::Buffer::Usage::STATIC_DRAW);
                ubo.unbind();
                ubo.bind_base(0);
            }
            Engine::Dispatcher().sink<Events::Callback::MouseCursor>().connect<&on_mouse_cursor>();
            Engine::Dispatcher().sink<Events::Callback::MouseScroll>().connect<&on_mouse_scroll>();
            Engine::Dispatcher().sink<Events::Callback::FramebufferResize>().connect<&on_framebuffer_resize>();
            Engine::Dispatcher().sink<Events::Key::F>().connect<&on_key_focus>();
            Engine::Dispatcher().sink<Events::Key::C>().connect<&on_key_center>();
        }
    }

    void ModuleCamera::begin_frame() {
        auto &camera = Engine::Context().get<Camera>();
        auto &keyboard = Engine::Context().get<Keyboard>();
        auto dt = PluginFrameTimer::delta();
        ViewParams v_params = GetViewParams(camera);
        glm::vec3 front = v_params.center - v_params.eye;
        if (keyboard.w()) {
            translate(camera, front * dt);
        }
        if (keyboard.s()) {
            translate(camera, -front * dt);
        }
        glm::vec3 right = glm::cross(front, v_params.up);
        if (keyboard.a()) {
            translate(camera, -right * dt);
        }
        if (keyboard.d()) {
            translate(camera, right * dt);
        }
    }

    void ModuleCamera::update() {
        auto &camera = Engine::Context().get<Camera>();
        auto &ubo = Engine::Context().get<CameraUniformBuffer>();
        ViewParams v_params = GetViewParams(camera);

        ubo.bind();
        if (camera.dirty_view) {
            ubo.buffer_sub_data(glm::value_ptr(camera.view), sizeof(camera.view));
            camera.dirty_view = false;
        }

        if (camera.dirty_proj) {
            ubo.buffer_sub_data(glm::value_ptr(camera.proj), sizeof(camera.proj), sizeof(camera.view));
            camera.dirty_proj = false;
        }
        ubo.unbind();
    }

    void ModuleCamera::end_frame() {
    }

    void ModuleCamera::deactivate() {
        if (base_deactivate()) {
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

        }
    }

    void ModuleCamera::render() {
    }

    void ModuleCamera::center_camera_at_distance(const Vector<float, 3> &center, float distance) {
        auto &camera = Engine::Context().get<Camera>();
        ViewParams v_params = GetViewParams(camera);
        PerspectiveParams p_params = GetPerspectiveParams(camera);

        glm::vec3 front = v_params.center - v_params.eye;
        v_params.center = center;
        v_params.eye = center - front * distance / tanf(p_params.fovy_degrees / 2.0f);
        SetViewParams(camera, v_params);
        fit_near_and_far_to_distance(distance);
    }

    void ModuleCamera::fit_near_and_far_to_distance(float distance) {
        auto &camera = Engine::Context().get<Camera>();

        if (camera.proj_type == Camera::ProjectionType::PERSPECTIVE) {
            PerspectiveParams p_params = GetPerspectiveParams(camera);
            p_params.zNear = distance / 100.0f;
            p_params.zFar = distance * 3.0f;
            SetPerspectiveParams(camera, p_params);
        } else if (camera.proj_type == Camera::ProjectionType::ORTHOGRAPHIC) {
            OrthoParams o_params = GetOrthoParams(camera);
            o_params.zNear = distance / 100.0f;
            o_params.zFar = distance * 3.0f;
            SetOrthoParams(camera, o_params);
        }
    }
}
