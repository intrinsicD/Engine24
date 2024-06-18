//
// Created by alex on 18.06.24.
//

#define GLAD_GL_IMPLEMENTATION

#include "PluginGlfwWindow.h"
#include "glad/gl.h"
#include "GLFW/glfw3.h"
#include "Logger.h"
#include "Engine.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

namespace Bcg {
    const GLuint WIDTH = 800, HEIGHT = 600;
    static bool show_window_gui = false;
    static float dpi = 1.5;
    float clear_color[3] = {0.2f, 0.3f, 0.3f};

    static void glfw_error_callback(int error, const char *description) {
        std::string message = "GLFW Error " + std::to_string(error) + ", " + description + "\n";
        Log::Error(message.c_str());
    }

// Is called whenever a key is pressed/released via GLFW
    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GL_TRUE);
    }

    static void resize_callback(GLFWwindow *window, int width, int height) {
        glViewport(0, 0, width, height);
    }

    static void close_callback(GLFWwindow *window) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }

    static void drop_callback(GLFWwindow *window, int count, const char **paths) {
        for (int i = 0; i < count; ++i) {
            Log::Info(paths[i]);
        }
    }
    static void load_fonts(ImGuiIO &io, float dpi) {
        io.Fonts->Clear();
        io.Fonts->AddFontFromFileTTF("../ext/imgui/misc/fonts/ProggyClean.ttf", 16.0f * dpi);
        io.Fonts->Build();
    }

    bool PluginGlfwWindow::init() {
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

        Engine::Instance()->window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", NULL, NULL);

        if (!Engine::Instance()->window) {
            Log::Error("Failed to create GLFW window");
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(Engine::Instance()->window);
        glfwSwapInterval(1);
        glfwSetWindowUserPointer(Engine::Instance()->window, Engine::Instance());

        // Set the required callback functions
        glfwSetKeyCallback(Engine::Instance()->window, key_callback);
        glfwSetWindowCloseCallback(Engine::Instance()->window, close_callback);
        glfwSetWindowSizeCallback(Engine::Instance()->window, resize_callback);
        glfwSetDropCallback(Engine::Instance()->window, drop_callback);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        auto &io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        io.IniFilename = nullptr;

        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(Engine::Instance()->window, true);
        ImGui_ImplOpenGL3_Init();

        load_fonts(io, dpi);
        ImGui::GetStyle().ScaleAllSizes(dpi);

        // Load OpenGL functions, gladLoadGL returns the loaded version, 0 on error.
        int version = gladLoadGL(glfwGetProcAddress);
        if (version == 0) {
            Log::Error("Failed to initialize OpenGL context");
            return false;
        }

        // Successfully loaded OpenGL
        std::string message = "Loaded OpenGL " + std::to_string(GLAD_VERSION_MAJOR(version)) + "." + std::to_string(GLAD_VERSION_MINOR(version));
        Log::Info(message.c_str());

        // Define the viewport dimensions
        glViewport(0, 0, WIDTH, HEIGHT);
        glClearColor(clear_color[0], clear_color[1], clear_color[2], 1.0f);
        return true;
    }

    void PluginGlfwWindow::clear_framebuffer(){
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void PluginGlfwWindow::start_gui(){
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::BeginMainMenuBar();
    }

    void PluginGlfwWindow::end_gui(){
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

    void PluginGlfwWindow::activate() {

    }

    void PluginGlfwWindow::update() {

    }

    void PluginGlfwWindow::deactivate() {

    }

    void PluginGlfwWindow::render_menu() {
        if (ImGui::BeginMenu("Plugin")) {
            ImGui::MenuItem("Window", nullptr, &show_window_gui);
            ImGui::EndMenu();
        }
    }

    void PluginGlfwWindow::render_gui() {
        if(show_window_gui){
            if(ImGui::Begin("Window", &show_window_gui, ImGuiWindowFlags_AlwaysAutoResize)){
                if(ImGui::InputFloat("Dpi", &dpi)){
                    Engine::Instance()->command_buffer.emplace_back([](){
                        load_fonts(ImGui::GetIO(), dpi);
                        ImGui::GetStyle().ScaleAllSizes(dpi);
                    });
                }
                int win_size[2];
                glfwGetWindowSize(glfwGetCurrentContext(), &win_size[0], &win_size[1]);
                if(ImGui::InputInt2("Size: ", &win_size[0])){
                    glfwSetWindowSize(Engine::Instance()->window, win_size[0], win_size[1]);
                }
                if(ImGui::ColorEdit3("Clear color", clear_color)){
                    glClearColor(clear_color[0], clear_color[1], clear_color[2], 1.0f);
                }
            }
            ImGui::End();
        }
    }

    void PluginGlfwWindow::render() {

    }
}