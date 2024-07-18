add_subdirectory(AABB)
add_subdirectory(Frustum)
add_subdirectory(Line)
add_subdirectory(OBB)
add_subdirectory(Plane)
add_subdirectory(Sphere)
add_subdirectory(Triangle)

target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        Intersections.cpp
)
