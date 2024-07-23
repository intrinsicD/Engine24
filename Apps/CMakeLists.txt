add_subdirectory(MeshViewer)
add_subdirectory(TestBed)
add_subdirectory(bgfx_test)

target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        Application.cpp
)