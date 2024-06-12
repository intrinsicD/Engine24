#include <iostream>
#define GLAD_GL_IMPLEMENTATION
#include "glad/gl.h"
#include "GLFW/glfw3.h"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"


// Function prototypes
GLFWwindow* create_window(const char *name, int major, int minor);
GladGLContext* create_context(GLFWwindow *window);
void free_context(GladGLContext *context);
void draw(GLFWwindow *window, GladGLContext *context, float r, float g, float b);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
void close_callback(GLFWwindow* window);

// Window dimensions
const GLuint WIDTH = 400, HEIGHT = 300;

int main()
{
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow *window1 = create_window("Window 1", 3, 3);
    GLFWwindow *window2 = create_window("Window 2", 3, 2);

    if (!window1 || !window2) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwSetKeyCallback(window1, key_callback);
    glfwSetKeyCallback(window2, key_callback);
    glfwSetWindowCloseCallback(window1, close_callback);
    glfwSetWindowCloseCallback(window2, close_callback);

    GladGLContext *context1 = create_context(window1);
    int version1 = gladLoadGLContext(context1, glfwGetProcAddress);
    GladGLContext *context2 = create_context(window2);
    int version2 = gladLoadGLContext(context2, glfwGetProcAddress);

    if (!context1 || !context2) {
        std::cout << "Failed to initialize GL contexts" << std::endl;
        free_context(context1);
        free_context(context2);
    }

    glfwMakeContextCurrent(window1);
    context1->Viewport(0, 0, WIDTH, HEIGHT);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window1, true);
    ImGui_ImplOpenGL3_Init("version 130");
    glfwMakeContextCurrent(window2);
    context2->Viewport(0, 0, WIDTH, HEIGHT);

    bool show_demo_window = true;


    while (!glfwWindowShouldClose(window1) || !glfwWindowShouldClose(window2))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);


        if (!glfwWindowShouldClose(window1)) {
            draw(window1, context1, 0.5, 0.2, 0.6);
        }

        if (!glfwWindowShouldClose(window2)) {
            draw(window2, context2, 0.0, 0.1, 0.8);
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    free_context(context1);
    free_context(context2);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window1);
    glfwDestroyWindow(window2);
    glfwTerminate();

    return 0;
}

GLFWwindow* create_window(const char *name, int major, int minor) {
    std::cout << "Creating Window, OpenGL " << major << "." << minor << ": " << name << std::endl;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, name, NULL, NULL);
    return window;
}

GladGLContext* create_context(GLFWwindow *window) {
    glfwMakeContextCurrent(window);

    GladGLContext* context = (GladGLContext*) calloc(1, sizeof(GladGLContext));
    if (!context) return NULL;

    int version = gladLoadGLContext(context, glfwGetProcAddress);
    std::cout << "Loaded OpenGL " << GLAD_VERSION_MAJOR(version) << "." << GLAD_VERSION_MINOR(version) << std::endl;

    return context;
}

void free_context(GladGLContext *context) {
    free(context);
}


void draw(GLFWwindow *window, GladGLContext *gl, float r, float g, float b) {
    glfwMakeContextCurrent(window);

    gl->ClearColor(r, g, b, 1.0f);
    gl->Clear(GL_COLOR_BUFFER_BIT);

    glfwSwapBuffers(window);
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

void close_callback(GLFWwindow *window) {
    glfwSetWindowShouldClose(window, GL_TRUE);
    glfwDestroyWindow(window);
}