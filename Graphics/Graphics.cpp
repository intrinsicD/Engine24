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
#include "config.h"
#include "EventsGui.h"
#include <filesystem>

namespace Bcg {

    struct Window{
        int WIDTH = 800, HEIGHT = 600;
        GLFWwindow *handle = nullptr;
        int version = 0;
        float dpi = 1.5;
        ImGuiContext *imgui_context = nullptr;
        bool show_window_gui = false;
        bool show_buffer_gui = false;
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

    static void resize_callback(GLFWwindow *window, int width, int height) {
        glViewport(0, 0, width, height);

        if (!ImGui::GetIO().WantCaptureMouse) {
            Engine::Dispatcher().trigger<Events::Callback::WindowResize>({window, width, height});
        }
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
        if (std::filesystem::exists(IMGUI_FONTS_PATH_APPS)) {
            io.Fonts->AddFontFromFileTTF((std::string(IMGUI_FONTS_PATH_APPS) + "/ProggyClean.ttf").c_str(),
                                         16.0f * dpi);
        } else {
            io.Fonts->AddFontFromFileTTF((std::string(IMGUI_FONTS_PATH) + "/ProggyClean.ttf").c_str(), 16.0f * dpi);
        }
        io.Fonts->Build();
    }

    bool Graphics::init() {
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

            global_window.handle = glfwCreateWindow(global_window.WIDTH, global_window.HEIGHT, "BCG_ENGINE", NULL, NULL);

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
            glfwSetWindowSizeCallback(global_window.handle, resize_callback);
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

            load_fonts(io, global_window.dpi);
            ImGui::GetStyle().ScaleAllSizes(global_window.dpi);
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
        Engine::Context().emplace<BufferContainer>();
        Engine::Dispatcher().trigger(Events::Callback::WindowResize{global_window.handle, global_window.WIDTH, global_window.HEIGHT});
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
            ImGui::MenuItem("Buffer", nullptr, &global_window.show_buffer_gui);
            ImGui::EndMenu();
        }
    }

    void Graphics::render_gui() {
        Engine::Dispatcher().trigger<Events::Gui::Render>();
        if (global_window.show_window_gui) {
            if (ImGui::Begin("Window", &global_window.show_window_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                if (ImGui::ColorEdit3("clear_color", global_window.clear_color)) {
                    glClearColor(global_window.clear_color[0], global_window.clear_color[1], global_window.clear_color[2], 1.0);
                }
                ImGui::Text("Width %d", global_window.WIDTH);
                ImGui::Text("Height %d", global_window.HEIGHT);
            }
            ImGui::End();
        }
        if (global_window.show_buffer_gui) {
            if (ImGui::Begin("Buffers", &global_window.show_window_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                render_gui(Engine::Context().get<BufferContainer>());
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

    Vector<int, 4> Graphics::get_viewport() {
        Vector<int, 4> viewport;
        glGetIntegerv(GL_VIEWPORT, viewport.data());
        return std::move(viewport);
    }

    bool Graphics::read_depth_buffer(int x, int y, float &zf) {
        Vector<int, 4> viewport = get_viewport();

        // in OpenGL y=0 is at the 'bottom'
        y = viewport[3] - y;

        // read depth buffer value at (x, y_new)
        glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &zf);
        return zf != 1.0f;
    }

    //------------------------------------------------------------------------------------------------------------------

    void Graphics::setup_batched_buffer(BatchedBuffer &batched_buffer) {
        if (batched_buffer.id == -1) {
            glGenBuffers(1, &batched_buffer.id);
        }
        glBindBuffer(batched_buffer.target, batched_buffer.id);
        int current_buffer_size;
        glGetBufferParameteriv(batched_buffer.target, GL_BUFFER_SIZE, &current_buffer_size);
        int required_buffer_size = batched_buffer.total_size_bytes();
        if (required_buffer_size != current_buffer_size) {
            glBufferData(batched_buffer.target, required_buffer_size, NULL, batched_buffer.usage);
        }

        for (const auto &item: batched_buffer.layout) {
            glBufferSubData(batched_buffer.target, item.offset, item.size_in_bytes, item.data);
        }
        glBindBuffer(batched_buffer.target, 0);
    }

    size_t Graphics::remove_buffer(const std::string &name) {
        auto &buffers = Engine::Context().get<BufferContainer>();
        return buffers.erase(name);
    }

    size_t Graphics::remove_buffer(unsigned int id) {
        auto &buffers = Engine::Context().get<BufferContainer>();
        size_t counter_erased = 0;
        for (auto &item: buffers) {
            if (item.second == id) {
                counter_erased += buffers.erase(item.first);
            }
        }
        return counter_erased;
    }

    size_t Graphics::buffer_size(unsigned int id, unsigned int target) {
        GLint buffer_size = 0;
        glBindBuffer(target, id);
        glGetBufferParameteriv(target, GL_BUFFER_SIZE, &buffer_size);
        return buffer_size;
    }

    unsigned int Graphics::get_or_add_buffer(const std::string &name) {
        auto &buffers = Engine::Context().get<BufferContainer>();
        auto iter = buffers.find(name);
        if (iter != buffers.end()) {
            return iter->second;
        }
        unsigned int id = 0;
        glGenBuffers(1, &id);
        if (id == 0) {
            Log::Error("OpenGL failed to generate a vaild buffer id!");
        } else {
            buffers[name] = id;
        }
        return id;
    }

    void Graphics::upload(unsigned int id, unsigned int target, const void *data, size_t size_bytes,
                          size_t offset) {
        glBindBuffer(target, id);
        if (offset > 0) {
            if (offset + size_bytes <= buffer_size(id, target)) {
                glBufferSubData(target, offset, size_bytes, data);
                return;
            }
        }
        glBufferData(target, size_bytes, data, GL_STATIC_DRAW);
    }

    void Graphics::upload_vbo(unsigned int id, const void *data, size_t size_bytes, size_t offset) {
        upload(id, GL_ARRAY_BUFFER, data, size_bytes, offset);
    }

    void Graphics::upload_ebo(unsigned int id, const void *data, size_t size_bytes, size_t offset) {
        upload(id, GL_ELEMENT_ARRAY_BUFFER, data, size_bytes, offset);
    }

    void Graphics::upload_ssbo(unsigned int id, const void *data, size_t size_bytes, size_t offset) {
        upload(id, GL_SHADER_STORAGE_BUFFER, data, size_bytes, offset);
    }

    void Graphics::upload_ubo(unsigned int id, const void *data, size_t size_bytes, size_t offset) {
        upload(id, GL_UNIFORM_BUFFER, data, size_bytes, offset);
    }

    void Graphics::render_gui(const BufferContainer &buffers) {
        for (const auto &item: buffers) {
            ImGui::Text("%s: %u", item.first.c_str(), item.second);
        }
    }

    //------------------------------------------------------------------------------------------------------------------

    static bool CheckCompileStatus(unsigned int shader_id) {
        int success;
        glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);

        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader_id, 512, nullptr, infoLog);
            Log::Error(("Shader Compilation Failed: " + std::string(infoLog)).c_str());
            return false;
        }
        return true;
    }

    static bool CheckLinkStatus(unsigned int program_id) {
        int success;
        glLinkProgram(program_id);
        // check for linking errors
        glGetProgramiv(program_id, GL_LINK_STATUS, &success);

        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(program_id, 512, nullptr, infoLog);

            Log::Error(("Program Linking Failed: " + std::string(infoLog)).c_str());
            return false;
        }
        return true;
    }

    static unsigned int create_shader(unsigned int type, const char *source) {
        unsigned int shader_id;
        shader_id = glCreateShader(type);
        glShaderSource(shader_id, 1, &source, nullptr);
        glCompileShader(shader_id);
        if (!CheckCompileStatus(shader_id)) {
            glDeleteShader(shader_id);
            shader_id = -1;
        }
        return shader_id;
    }

    unsigned int Graphics::create_program(const char *vs_source, const char *fs_source,
                                          const char *gs_source, const char *tc_source,
                                          const char *te_source) {
        if (!vs_source) return -1;
        if (!fs_source) return -1;

        auto vertex_shader = create_shader(GL_VERTEX_SHADER, vs_source);
        auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fs_source);

        if (vertex_shader == -1 | fragment_shader == -1) {
            return -1;
        }

        auto program = glCreateProgram();

        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);

        unsigned int geometry_shader = -1;
        if (gs_source) {
            geometry_shader = create_shader(GL_GEOMETRY_SHADER, gs_source);
            if (geometry_shader != -1) {
                glAttachShader(program, geometry_shader);
            }
        }

        unsigned int tc_shader = -1;
        if (tc_source) {
            tc_shader = create_shader(GL_TESS_CONTROL_SHADER, tc_source);
            if (tc_shader != -1) {
                glAttachShader(program, tc_shader);
            }
        }

        unsigned int te_shader = -1;
        if (te_source) {
            te_shader = create_shader(GL_TESS_EVALUATION_SHADER, te_source);
            if (te_shader != -1) {
                glAttachShader(program, te_shader);
            }
        }

        glLinkProgram(program);

        if (!CheckLinkStatus(program)) {
            glDeleteProgram(program);
            program = -1;
        }

        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        if (geometry_shader) {
            glDeleteShader(geometry_shader);
        }
        if (tc_shader) {
            glDeleteShader(tc_shader);
        }
        if (te_shader) {
            glDeleteShader(te_shader);
        }
        return program;
    }

    static std::string ReadTextFile(const char *path) {
        if (!path) {
            return "";
        }
        std::string result_text;
        long length;
        FILE *file = fopen(path, "rb");

        if (file) {
            fseek(file, 0, SEEK_END);
            length = ftell(file);
            fseek(file, 0, SEEK_SET);
            result_text.resize(length);
            if (static_cast<long>(result_text.size()) == length) {
                auto result = fread(result_text.data(), 1, length, file);
                if (result != length) return "";
            }
            fclose(file);
        }
        return result_text;
    }

    std::string Graphics::load_shader(const char *filepath) {
        return ReadTextFile(filepath);
    }

    //------------------------------------------------------------------------------------------------------------------




    Shader::Shader(unsigned int type) : type(type), id(-1) {}

    Shader Shader::VertexShader() {
        return {GL_VERTEX_SHADER};
    }

    Shader Shader::FragmentShader() {
        return {GL_FRAGMENT_SHADER};
    }

    Shader Shader::GeometryShader() {
        return {GL_GEOMETRY_SHADER};
    }

    Shader Shader::TessContrlShader() {
        return {GL_TESS_CONTROL_SHADER};
    }

    Shader Shader::TessEvalShader() {
        return {GL_TESS_EVALUATION_SHADER};
    }

    Shader Shader::ComputeShader() {
        return {GL_COMPUTE_SHADER};
    }

    bool Shader::load(const char *path) {
        source = ReadTextFile(path).c_str();
        this->path = path;
        return compile();
    }

    bool Shader::compile() {
        if (!source) { return false; }
        id = glCreateShader(type);
        glShaderSource(id, 1, &source, nullptr);
        glCompileShader(id);
        if (!CheckCompileStatus(id)) {
            glDeleteShader(id);
            id = -1;
            return false;
        }
        return true;
    }

    Program::Program(const char *name) : id(-1), name(name) {}

    bool Program::load(const char *v_path,
                       const char *f_path,
                       const char *g_path,
                       const char *tc_path,
                       const char *te_path) {
        Shader vs = Shader::VertexShader();
        if (!vs.load(v_path)) {
            return false;
        }
        Shader fs = Shader::FragmentShader();
        if (!fs.load(f_path)) {
            return false;
        }
        Shader gs = Shader::GeometryShader();
        gs.load(g_path);
        Shader tcs = Shader::TessContrlShader();
        tcs.load(tc_path);
        Shader tes = Shader::TessEvalShader();
        tes.load(te_path);
        return link(vs, fs, gs, tcs, tes);
    }

    bool Program::load(const char *c_path) {
        Shader cs = Shader::ComputeShader();
        if (!cs.load(c_path)) {
            return false;
        }
        return link(cs);
    }

    bool Program::link(const Shader &vs,
                       const Shader &fs,
                       const Shader &gs,
                       const Shader &tcs,
                       const Shader &tes) {
        if (!vs || !fs) {
            Log::Error("Program requires at least a vertex_shader and fragment_shader!");
            return false;
        }
        if (vs) {
            shaders.emplace_back(vs);
        }
        if (fs) {
            shaders.emplace_back(fs);
        }
        if (gs) {
            shaders.emplace_back(gs);
        }
        if (tcs) {
            shaders.emplace_back(tcs);
        }
        if (tes) {
            shaders.emplace_back(tes);
        }
        return link();
    }

    bool Program::link(const Shader &cs) {
        if (!cs) return false;
        shaders.emplace_back(cs);
        return link();
    }


    bool Program::link() {
        id = glCreateProgram();
        for (const auto &shader: shaders) {
            glAttachShader(id, shader.id);
        }
        glLinkProgram(id);
        if (!CheckLinkStatus(id)) {
            glDeleteProgram(id);
            id = -1;
            return false;
        }
        for (const auto &shader: shaders) {
            glDeleteShader(shader.id);
        }
        return true;
    }
}