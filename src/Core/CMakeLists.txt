target_sources(Engine24 PRIVATE
        CameraCommands.cpp
        Color.cpp
        CommandBuffer.cpp
        Engine.cpp
        Entity.cpp
        EntityCommands.cpp
        FileWatcher.cpp
        HierarchyCommands.cpp
        Input.cpp
        KDTreeCompute.cpp
        KDTreeCpu.cpp
        KDTreeCuda.cu
        Keyboard.cpp
        Logger.cpp
        Mouse.cpp
        Utils.cpp
)

add_subdirectory(Guis)
add_subdirectory(Plugins)