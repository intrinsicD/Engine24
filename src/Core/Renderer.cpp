//
// Created by alex on 6/16/25.
//

#include "Renderer.h"
#include "Logger.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>

// ImGui headers
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <ImGuizmo/ImGuizmo.h>

#include "Engine.h"
#include "Events/EventsGui.h"

namespace Bcg {
    Renderer::Renderer(Window &window) : m_window(window) {
        init_graphics();
        init_imgui();
    }

    Renderer::~Renderer() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    void Renderer::begin_frame() {
        // In a Vulkan engine, this is where you would acquire the next swapchain image.
        // In OpenGL, it can be a good place to poll for state or start profiling.

        // Set the viewport based on the window's framebuffer size
        glm::ivec2 fb_size = m_window.get_framebuffer_size();
        glViewport(0, 0, fb_size.x, fb_size.y);

        set_clear_color(m_clear_color);
        clear_framebuffer();
    }

    void Renderer::end_frame() {
        m_window.swap_buffers();
    }

    void Renderer::set_clear_color(const Vector<float, 4> &color) {
        m_clear_color = color;
    }

    const Vector<float, 4> &Renderer::get_clear_color() const {
        return m_clear_color;
    }

    void Renderer::clear_framebuffer() {
        glClearColor(m_clear_color.r, m_clear_color.g, m_clear_color.b, m_clear_color.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void Renderer::begin_gui() {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();
        ImGui::BeginMainMenuBar();
        Engine::Dispatcher().trigger<Events::Gui::Menu::Render>();
        Engine::Dispatcher().trigger<Events::Gui::Render>();
    }

    void Renderer::end_gui() {
        ImGui::EndMainMenuBar();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        ImGuiIO &io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            GLFWwindow *backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
    }

    void Renderer::init_graphics() {
        // Make the window's context current
        glfwMakeContextCurrent(m_window.get_native_window());

        int version = gladLoadGL(glfwGetProcAddress);
        // Initialize GLAD (or your OpenGL function loader)
        if (version == 0) {
            Log::Error("Failed to initialize GLAD");
            // Handle error...
            return;
        }

        Log::Info("  Vendor: " + std::string((const char *) glGetString(GL_VENDOR)));
        Log::Info("  Renderer: " + std::string((const char *) glGetString(GL_RENDERER)));
        Log::Info("  Version: " + std::string((const char *) glGetString(GL_VERSION)));

        // Set default OpenGL state
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CCW);

        //TODO Where should i put this?
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    static void load_fonts(ImGuiIO &io, float dpi) {
        io.Fonts->Clear();
        io.Fonts->AddFontFromFileTTF("../Fonts/ProggyClean.ttf", 16.0f * dpi);
        io.Fonts->Build();
    }

    void Renderer::init_imgui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

        ImGui::StyleColorsDark();

        ImGuiStyle &style = ImGui::GetStyle();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(m_window.get_native_window(), true);
        ImGui_ImplOpenGL3_Init();

        // Load fonts and scale styles based on DPI
        float dpi = m_window.get_xy_dpi_scaling()[0];
        load_fonts(io, dpi);
        ImGui::GetStyle().ScaleAllSizes(dpi);
    }

    Vector<float, 4> Renderer::get_viewport() const {
        Vector<float, 4> viewport;
        glGetFloatv(GL_VIEWPORT, glm::value_ptr(viewport));
        return viewport;
    }

    Vector<float, 4> Renderer::get_viewport_dpi_adjusted() const {
        Vector<int, 4> vp = get_viewport();
        return vp * int(m_window.get_xy_dpi_scaling()[0]);
    }
}
