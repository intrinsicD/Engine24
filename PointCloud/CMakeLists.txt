target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        PointCloud.cpp
        PointCloudGui.cpp
        PluginPointCloud.cpp
        PointCloudCommands.cpp
        PointCloudCompute.cpp
        io/read_pts.cpp
        io/read_xyz.cpp
        io/read_csv.cpp
)