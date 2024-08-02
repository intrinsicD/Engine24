//
// Created by alex on 19.06.24.
//
#define GLAD_GL_IMPLEMENTATION

#include "Graphics.h"
#include "Engine.h"
#include "EventsCallbacks.h"
#include "Logger.h"
#include "Input.h"
#include "HandleGlfwKeyEvents.h"
#include "glad/gl.h"
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "EventsGui.h"

namespace Bcg {

    struct Window {
        const char *title = nullptr;
        int WIDTH = 800, HEIGHT = 600;
        GLFWwindow *handle = nullptr;
        int version = 0;
        ImGuiContext *imgui_context = nullptr;
        bool show_window_gui = false;
        float clear_color[3] = {0.2f, 0.3f, 0.3f};
    };

    static Window global_window;

    static void glfw_error_callback(int error, const char *description) {
        std::string message = "GLFW Error " + std::to_string(error) + ", " + description + "\n";
        Log::Error(message.c_str());
    }

    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
        auto &keyboard = Input::set_keyboard(window, key, scancode, action, mode);

        if (!keyboard.gui_captured) {
            auto &dispatcher = Engine::Dispatcher();

            dispatcher.trigger<Events::Callback::Key>({window, key, scancode, action, mode});
            if (keyboard.esc()) {
                glfwSetWindowShouldClose(window, true);
            }
            handle(key, action, dispatcher);
        }
    }

    static void mouse_cursor_callback(GLFWwindow *window, double xpos, double ypos) {
        Input::set_mouse_cursor_position(window, xpos, ypos);
        //TODO figure out how to either control the camera or on strg space control the selected object...

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::MouseCursor>({window, xpos, ypos});
        }
    }

    static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
        Input::set_mouse_button(window, button, action, mods);

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::MouseButton>({window, button, action, mods});
        }
    }

    static void mouse_scrolling(GLFWwindow *window, double xoffset, double yoffset) {
        Input::set_mouse_scrolling(window, xoffset, yoffset);

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::MouseScroll>({window, xoffset, yoffset});
        }
    }

    static void window_resize_callback(GLFWwindow *window, int width, int height) {
        Engine::Dispatcher().trigger(Events::Callback::WindowResize{global_window.handle, width, height});
    }

    static void framebuffer_resize_callback(GLFWwindow *window, int width, int height) {
        glViewport(0, 0, width, height);
        Engine::Dispatcher().trigger(Events::Callback::FramebufferResize{global_window.handle, width, height});
    }

    static void close_callback(GLFWwindow *window) {
        glfwSetWindowShouldClose(window, true);

        Engine::Dispatcher().trigger<Events::Callback::WindowClose>({window});
    }

    static void drop_callback(GLFWwindow *window, int count, const char **paths) {
        for (int i = 0; i < count; ++i) {
            Log::Info("Dropped: " + std::string(paths[i]));
        }

        Engine::Dispatcher().trigger<Events::Callback::Drop>({window, count, paths});
    }

    static void load_fonts(ImGuiIO &io, float dpi) {
        io.Fonts->Clear();
        io.Fonts->AddFontFromFileTTF("../Fonts/ProggyClean.ttf", 16.0f * dpi);
        io.Fonts->Build();
    }

    bool Graphics::init(int width, int height, const char *title) {
        if (global_window.handle) {
            Log::Info("GLFW context already initialized");
        } else {
            glfwSetErrorCallback(glfw_error_callback);
            Log::Info("Starting GLFW context, OpenGL 4.6");
            // Init GLFW
            if (!glfwInit()) {
                Log::Error("Failed to initialize GLFW context");
                return false;
            }

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

            global_window.WIDTH = width;
            global_window.HEIGHT = height;
            global_window.title = title;
            global_window.handle = glfwCreateWindow(global_window.WIDTH, global_window.HEIGHT, global_window.title,
                                                    NULL, NULL);

            if (!global_window.handle) {
                Log::Error("Failed to create GLFW window");
                glfwTerminate();
                return false;
            }

            glfwMakeContextCurrent(global_window.handle);
            glfwSwapInterval(1);
            glfwSetWindowUserPointer(global_window.handle, Engine::Instance());

            // Set the required callback functions
            glfwSetKeyCallback(global_window.handle, key_callback);
            glfwSetCursorPosCallback(global_window.handle, mouse_cursor_callback);
            glfwSetMouseButtonCallback(global_window.handle, mouse_button_callback);
            glfwSetScrollCallback(global_window.handle, mouse_scrolling);
            glfwSetWindowCloseCallback(global_window.handle, close_callback);
            glfwSetWindowSizeCallback(global_window.handle, window_resize_callback);
            glfwSetFramebufferSizeCallback(global_window.handle, framebuffer_resize_callback);
            glfwSetDropCallback(global_window.handle, drop_callback);
        }

        // Load OpenGL functions, gladLoadGL returns the loaded version, 0 on error.
        if (global_window.version != 0) {
            Log::Info("OpenGL context already initialized");
        } else {
            global_window.version = gladLoadGL(glfwGetProcAddress);
            if (global_window.version == 0) {
                Log::Error("Failed to initialize OpenGL context");
                return false;
            }

            // Successfully loaded OpenGL
            std::string message = "Loaded OpenGL " + std::to_string(GLAD_VERSION_MAJOR(global_window.version)) + "." +
                                  std::to_string(GLAD_VERSION_MINOR(global_window.version));
            Log::Info(message.c_str());
        }

        if (global_window.imgui_context) {
            Log::Info("ImGui Context already initialized");
        } else {
            IMGUI_CHECKVERSION();
            global_window.imgui_context = ImGui::CreateContext();
            auto &io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
            io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
            io.IniFilename = nullptr;

            ImGui::StyleColorsDark();
            ImGui_ImplGlfw_InitForOpenGL(glfwGetCurrentContext(), true);
            ImGui_ImplOpenGL3_Init();

            float dpi = dpi_scaling();

            load_fonts(io, dpi);
            ImGui::GetStyle().ScaleAllSizes(dpi);
        }
        glViewport(0, 0, global_window.WIDTH, global_window.HEIGHT);
        glClearColor(global_window.clear_color[0], global_window.clear_color[1], global_window.clear_color[2], 1.0f);
        // Enable depth testing
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS); // Default depth function
        // Enable face culling
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK); // Cull back faces
        glFrontFace(GL_CCW); // Counter-clockwise front faces
// Enable blending for transparency
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_PROGRAM_POINT_SIZE);
        Vector<int, 2> fbs = Graphics::get_framebuffer_size();
        Engine::Dispatcher().enqueue(Events::Callback::FramebufferResize{global_window.handle, fbs.x(), fbs.y()});
        return true;
    }

    bool Graphics::should_close() {
        return glfwWindowShouldClose(global_window.handle);
    }

    void Graphics::poll_events() {
        glfwPollEvents();
    }

    void Graphics::set_window_title(const char *title) {
        glfwSetWindowTitle(global_window.handle, title);
    }

    void Graphics::set_clear_color(const float *color) {
        *global_window.clear_color = *color;
        glClearColor(global_window.clear_color[0], global_window.clear_color[1], global_window.clear_color[2], 1.0f);
    }

    void Graphics::clear_framebuffer() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void Graphics::start_gui() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();

        ImGui::BeginMainMenuBar();
    }

    void Graphics::render_menu() {
        Engine::Dispatcher().trigger<Events::Gui::Menu::Render>();
        if (ImGui::BeginMenu("Graphics")) {
            ImGui::MenuItem("Window", nullptr, &global_window.show_window_gui);
            ImGui::EndMenu();
        }
    }

    void Graphics::render_gui() {
        Engine::Dispatcher().trigger<Events::Gui::Render>();
        if (global_window.show_window_gui) {
            if (ImGui::Begin("Window", &global_window.show_window_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                if (ImGui::ColorEdit3("clear_color", global_window.clear_color)) {
                    glClearColor(global_window.clear_color[0], global_window.clear_color[1],
                                 global_window.clear_color[2], 1.0);
                }
                ImGui::Text("Width %d", global_window.WIDTH);
                ImGui::Text("Height %d", global_window.HEIGHT);
            }
            ImGui::End();
        }
    }

    void Graphics::end_gui() {
        ImGui::EndMainMenuBar();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        auto &io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
    }

    void Graphics::swap_buffers() {
        glfwSwapBuffers(global_window.handle);
    }

    Vector<int, 2> Graphics::get_window_pos() {
        int windowPosX, windowPosY;
        glfwGetWindowPos(global_window.handle, &windowPosX, &windowPosY);
        return {windowPosX, windowPosY};
    }

    Vector<int, 2> Graphics::get_window_size() {
        int width, height;
        glfwGetWindowSize(global_window.handle, &width, &height);
        return {width, height};
    }

    Vector<int, 2> Graphics::get_framebuffer_size() {
        int width, height;
        glfwGetFramebufferSize(global_window.handle, &width, &height);
        return {width, height};
    }

    Vector<int, 4> Graphics::get_viewport() {
        Vector<int, 4> viewport;
        glGetIntegerv(GL_VIEWPORT, viewport.data());
        return std::move(viewport);
    }

    Vector<int, 4> Graphics::get_viewport_dpi_adjusted() {
        Vector<int, 4> vp = get_viewport();
        return vp * int(dpi_scaling());
    }

    bool Graphics::read_depth_buffer(int x, int y, float &zf) {
        Vector<int, 4> viewport = get_viewport();

        // in OpenGL y=0 is at the 'bottom'
        y = viewport[3] - y;

        // read depth buffer value at (x, y_new)
        glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &zf);
        return zf != 1.0f;
    }

    float Graphics::dpi_scaling() {
        float dpi_scaling_factor;
        glfwGetWindowContentScale(global_window.handle, &dpi_scaling_factor, &dpi_scaling_factor);
        return dpi_scaling_factor;
    }

    //------------------------------------------------------------------------------------------------------------------
}