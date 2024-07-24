add_executable(bgfx_test)

target_link_libraries(bgfx_test PRIVATE Engine24 bgfx bx bimg example-common glfw)
target_sources(bgfx_test PRIVATE
        main.cpp BGFXApplication.cpp
        #TODO bgfx comes with a reduced subset of imgui, so i cannot just include all of imgui... 
)