target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        PluginCuda.cpp
)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
set(CUDAToolkit_ROOT /usr/local/cuda)
find_package(CUDAToolkit REQUIRED)

target_include_directories(Engine24 PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(Engine24 ${CUDAToolkit_LIBRARIES})
message("CUDAToolkit Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")