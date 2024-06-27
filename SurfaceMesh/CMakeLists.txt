target_include_directories(Engine24 SYSTEM PUBLIC .)
target_sources(Engine24 PRIVATE
    SurfaceMesh.cpp
        io/io.cpp
        io/read_stl.cpp
        io/read_pmp.cpp
        io/read_off.cpp
        io/read_obj.cpp
        io/write_stl.cpp
        io/write_pmp.cpp
        io/write_off.cpp
        io/write_obj.cpp
)