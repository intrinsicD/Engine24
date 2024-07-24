add_executable(TestBed main.cpp OpenGLApplication.cpp)

target_link_libraries(TestBed PUBLIC Engine24 imgui imguizmo ImGuiFileDialog)