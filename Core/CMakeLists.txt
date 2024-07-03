target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        Engine.cpp
        FileWatcher.cpp
        Graphics.cpp
        Input.cpp
        Keyboard.cpp
        Logger.cpp
        Materials.cpp
        Mouse.cpp
        Picker.cpp
        Plugin.cpp
        Plugins.cpp
)