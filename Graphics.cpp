//
// Created by alex on 19.06.24.
//
#define GLAD_GL_IMPLEMENTATION

#include "Graphics.h"
#include "Engine.h"
#include "EventsCallbacks.h"
#include "Logger.h"
#include "Keybaord.h"
#include "Mouse.h"
#include "glad/gl.h"
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

namespace Bcg {
    const GLuint WIDTH = 800, HEIGHT = 600;
    static GLFWwindow *global_window = nullptr;
    static int version = 0;
    static float dpi = 1.5;
    static ImGuiContext *imgui_context = nullptr;
    static bool show_window_gui = false;
    float clear_color[3] = {0.2f, 0.3f, 0.3f};

    const char *KeyName(int key);

    static void glfw_error_callback(int error, const char *description) {
        std::string message = "GLFW Error " + std::to_string(error) + ", " + description + "\n";
        Log::Error(message.c_str());
    }

// Is called whenever a key is pressed/released via GLFW
    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
        auto &keyboard = Engine::Context().get<Keyboard>();
        const char *name = KeyName(key);
        if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
            keyboard.shift = {name, key, scancode, action, mode};
        } else if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT) {
            keyboard.alt = {name, key, scancode, action, mode};
        } else if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
            keyboard.strg = {name, key, scancode, action, mode};
        } else if (key == GLFW_KEY_ESCAPE) {
            keyboard.esc = {name, key, scancode, action, mode};
        }
        keyboard.pressed[key] = action;

        Engine::Dispatcher().trigger<Events::Callback::Key>({window, key, scancode, action, mode});
        if (keyboard.esc) {
            glfwSetWindowShouldClose(window, true);
        }
    }

    static void mouse_cursor_callback(GLFWwindow *window, double xpos, double ypos) {
        auto &mouse = Engine::Context().get<Mouse>();
        mouse.cursor = {xpos, ypos};

        Engine::Dispatcher().trigger<Events::Callback::MouseCursor>({window, xpos, ypos});
    }

    static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
        auto &mouse = Engine::Context().get<Mouse>();
        if (button == GLFW_MOUSE_BUTTON_1) {
            mouse.left = {button, action, mods};
        } else if (button == GLFW_MOUSE_BUTTON_2) {
            mouse.right = {button, action, mods};
        } else if (button == GLFW_MOUSE_BUTTON_3) {
            mouse.middle = {button, action, mods};
        }

        Engine::Dispatcher().trigger<Events::Callback::MouseButton>({window, button, action, mods});
    }

    static void resize_callback(GLFWwindow *window, int width, int height) {
        glViewport(0, 0, width, height);

        Engine::Dispatcher().trigger<Events::Callback::WindowResize>({window, width, height});
    }

    static void close_callback(GLFWwindow *window) {
        glfwSetWindowShouldClose(window, true);

        Engine::Dispatcher().trigger<Events::Callback::WindowClose>({window});
    }

    static void drop_callback(GLFWwindow *window, int count, const char **paths) {
        for (int i = 0; i < count; ++i) {
            Log::Info(paths[i]);
        }

        Engine::Dispatcher().trigger<Events::Callback::Drop>({window, count, paths});
    }

    static void load_fonts(ImGuiIO &io, float dpi) {
        io.Fonts->Clear();
        io.Fonts->AddFontFromFileTTF("../ext/imgui/misc/fonts/ProggyClean.ttf", 16.0f * dpi);
        io.Fonts->Build();
    }

    bool Graphics::init() {
        if (global_window) {
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

            global_window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", NULL, NULL);

            if (!global_window) {
                Log::Error("Failed to create GLFW window");
                glfwTerminate();
                return false;
            }

            glfwMakeContextCurrent(global_window);
            glfwSwapInterval(1);
            glfwSetWindowUserPointer(global_window, Engine::Instance());

            // Set the required callback functions
            glfwSetKeyCallback(global_window, key_callback);
            glfwSetCursorPosCallback(global_window, mouse_cursor_callback);
            glfwSetMouseButtonCallback(global_window, mouse_button_callback);
            glfwSetWindowCloseCallback(global_window, close_callback);
            glfwSetWindowSizeCallback(global_window, resize_callback);
            glfwSetDropCallback(global_window, drop_callback);
        }

        // Load OpenGL functions, gladLoadGL returns the loaded version, 0 on error.
        if (version != 0) {
            Log::Info("OpenGL context already initialized");
        } else {
            version = gladLoadGL(glfwGetProcAddress);
            if (version == 0) {
                Log::Error("Failed to initialize OpenGL context");
                return false;
            }

            // Successfully loaded OpenGL
            std::string message = "Loaded OpenGL " + std::to_string(GLAD_VERSION_MAJOR(version)) + "." +
                                  std::to_string(GLAD_VERSION_MINOR(version));
            Log::Info(message.c_str());
        }

        if (imgui_context) {
            Log::Info("ImGui Context already initialized");
        } else {
            IMGUI_CHECKVERSION();
            imgui_context = ImGui::CreateContext();
            auto &io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
            io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
            io.IniFilename = nullptr;

            ImGui::StyleColorsDark();
            ImGui_ImplGlfw_InitForOpenGL(glfwGetCurrentContext(), true);
            ImGui_ImplOpenGL3_Init();

            load_fonts(io, dpi);
            ImGui::GetStyle().ScaleAllSizes(dpi);
        }

        glViewport(0, 0, WIDTH, HEIGHT);
        glClearColor(clear_color[0], clear_color[1], clear_color[2], 1.0f);
        return true;
    }

    bool Graphics::should_close() const {
        return glfwWindowShouldClose(global_window);
    }

    void Graphics::poll_events() const {
        glfwPollEvents();
    }

    void Graphics::set_clear_color(const float *color) {
        glClearColor(clear_color[0], clear_color[1], clear_color[2], 1.0f);
    }

    void Graphics::clear_framebuffer() const {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    void Graphics::start_gui() const {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::BeginMainMenuBar();
    }

    void Graphics::render_menu() const {
        if (ImGui::BeginMenu("Graphics")) {
            ImGui::MenuItem("Window", nullptr, &show_window_gui);
            ImGui::EndMenu();
        }
    }

    void Graphics::render_gui() const {
        if (show_window_gui) {
            if (ImGui::Begin("Window", &show_window_gui, ImGuiWindowFlags_AlwaysAutoResize)) {
                if (ImGui::ColorEdit3("clear_color", clear_color)) {
                    glClearColor(clear_color[0], clear_color[1], clear_color[2], 1.0);
                }
            }
            ImGui::End();
        }
    }

    void Graphics::end_gui() const {
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

    void Graphics::swap_buffers() const {
        glfwSwapBuffers(Bcg::global_window);
    }

    const char *KeyName(int key) {
        switch (key) {
            case GLFW_KEY_SPACE: return " ";
            case GLFW_KEY_APOSTROPHE: return "'";
            case GLFW_KEY_COMMA: return ",";
            case GLFW_KEY_MINUS: return "-";
            case GLFW_KEY_PERIOD: return ".";
            case GLFW_KEY_SLASH: return "/";
            case GLFW_KEY_0: return "0";
            case GLFW_KEY_1: return "1";
            case GLFW_KEY_2: return "2";
            case GLFW_KEY_3: return "3";
            case GLFW_KEY_4: return "4";
            case GLFW_KEY_5: return "5";
            case GLFW_KEY_6: return "6";
            case GLFW_KEY_7: return "7";
            case GLFW_KEY_8: return "8";
            case GLFW_KEY_9: return "9";
            case GLFW_KEY_SEMICOLON: return ";";
            case GLFW_KEY_EQUAL: return "=";
            case GLFW_KEY_A: return "A";
            case GLFW_KEY_B: return "B";
            case GLFW_KEY_C: return "C";
            case GLFW_KEY_D: return "D";
            case GLFW_KEY_E: return "E";
            case GLFW_KEY_F: return "F";
            case GLFW_KEY_G: return "G";
            case GLFW_KEY_H: return "H";
            case GLFW_KEY_I: return "I";
            case GLFW_KEY_J: return "J";
            case GLFW_KEY_K: return "K";
            case GLFW_KEY_L: return "L";
            case GLFW_KEY_M: return "M";
            case GLFW_KEY_N: return "N";
            case GLFW_KEY_O: return "O";
            case GLFW_KEY_P: return "P";
            case GLFW_KEY_Q: return "Q";
            case GLFW_KEY_R: return "R";
            case GLFW_KEY_S: return "S";
            case GLFW_KEY_T: return "T";
            case GLFW_KEY_U: return "U";
            case GLFW_KEY_V: return "V";
            case GLFW_KEY_W: return "W";
            case GLFW_KEY_X: return "X";
            case GLFW_KEY_Y: return "Y";
            case GLFW_KEY_Z: return "Z";
            case GLFW_KEY_LEFT_BRACKET: return "[";
            case GLFW_KEY_BACKSLASH: return "\\";
            case GLFW_KEY_RIGHT_BRACKET: return "]";
            case GLFW_KEY_GRAVE_ACCENT: return "`";
            case GLFW_KEY_WORLD_1: return "WORLD_1";
            case GLFW_KEY_WORLD_2: return "WORLD_2";
            case GLFW_KEY_ESCAPE: return "ESCAPE";
            case GLFW_KEY_ENTER: return "ENTER";
            case GLFW_KEY_TAB: return "TAB";
            case GLFW_KEY_BACKSPACE: return "BACKSPACE";
            case GLFW_KEY_INSERT: return "INSERT";
            case GLFW_KEY_DELETE: return "DELETE";
            case GLFW_KEY_RIGHT: return "RIGHT";
            case GLFW_KEY_LEFT: return "LEFT";
            case GLFW_KEY_DOWN: return "DOWN";
            case GLFW_KEY_UP: return "UP";
            case GLFW_KEY_PAGE_UP: return "PAGE_UP";
            case GLFW_KEY_PAGE_DOWN: return "PAGE_DOWN";
            case GLFW_KEY_HOME: return "HOME";
            case GLFW_KEY_END: return "END";
            case GLFW_KEY_CAPS_LOCK: return "CAPS_LOCK";
            case GLFW_KEY_SCROLL_LOCK: return "SCROLL_LOCK";
            case GLFW_KEY_NUM_LOCK: return "NUM_LOCK";
            case GLFW_KEY_PRINT_SCREEN: return "PRINT_SCREEN";
            case GLFW_KEY_PAUSE: return "PAUSE";
            case GLFW_KEY_F1: return "F1";
            case GLFW_KEY_F2: return "F2";
            case GLFW_KEY_F3: return "F3";
            case GLFW_KEY_F4: return "F4";
            case GLFW_KEY_F5: return "F5";
            case GLFW_KEY_F6: return "F6";
            case GLFW_KEY_F7: return "F7";
            case GLFW_KEY_F8: return "F8";
            case GLFW_KEY_F9: return "F9";
            case GLFW_KEY_F10: return "F10";
            case GLFW_KEY_F11: return "F11";
            case GLFW_KEY_F12: return "F12";
            case GLFW_KEY_F13: return "F13";
            case GLFW_KEY_F14: return "F14";
            case GLFW_KEY_F15: return "F15";
            case GLFW_KEY_F16: return "F16";
            case GLFW_KEY_F17: return "F17";
            case GLFW_KEY_F18: return "F18";
            case GLFW_KEY_F19: return "F19";
            case GLFW_KEY_F20: return "F20";
            case GLFW_KEY_F21: return "F21";
            case GLFW_KEY_F22: return "F22";
            case GLFW_KEY_F23: return "F23";
            case GLFW_KEY_F24: return "F24";
            case GLFW_KEY_F25: return "F25";
            case GLFW_KEY_KP_0: return "KP_0";
            case GLFW_KEY_KP_1: return "KP_1";
            case GLFW_KEY_KP_2: return "KP_2";
            case GLFW_KEY_KP_3: return "KP_3";
            case GLFW_KEY_KP_4: return "KP_4";
            case GLFW_KEY_KP_5: return "KP_5";
            case GLFW_KEY_KP_6: return "KP_6";
            case GLFW_KEY_KP_7: return "KP_7";
            case GLFW_KEY_KP_8: return "KP_8";
            case GLFW_KEY_KP_9: return "KP_9";
            case GLFW_KEY_KP_DECIMAL: return "KP_DECIMAL";
            case GLFW_KEY_KP_DIVIDE: return "KP_DIVIDE";
            case GLFW_KEY_KP_MULTIPLY: return "KP_MULTIPLY";
            case GLFW_KEY_KP_SUBTRACT: return "KP_SUBTRACT";
            case GLFW_KEY_KP_ADD: return "KP_ADD";
            case GLFW_KEY_KP_ENTER: return "KP_ENTER";
            case GLFW_KEY_KP_EQUAL: return "KP_EQUAL";
            case GLFW_KEY_LEFT_SHIFT: return "LEFT_SHIFT";
            case GLFW_KEY_LEFT_CONTROL: return "LEFT_CONTROL";
            case GLFW_KEY_LEFT_ALT: return "LEFT_ALT";
            case GLFW_KEY_LEFT_SUPER: return "LEFT_SUPER";
            case GLFW_KEY_RIGHT_SHIFT: return "RIGHT_SHIFT";
            case GLFW_KEY_RIGHT_CONTROL: return "RIGHT_CONTROL";
            case GLFW_KEY_RIGHT_ALT: return "RIGHT_ALT";
            case GLFW_KEY_RIGHT_SUPER: return "RIGHT_SUPER";
            case GLFW_KEY_MENU: return "MENU";
            default: return "UNKNOWN";
        }
    }
}