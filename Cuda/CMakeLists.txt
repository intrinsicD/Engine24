target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        PluginCuda.cpp
        KDTreeCuda.cu
)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

target_include_directories(Engine24 PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(Engine24 ${CUDAToolkit_LIBRARIES})

set(CMAKE_CUDA_ARCHITECTURES native)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message("CUDAToolkit Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")
target_compile_options(Engine24 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(Engine24 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(Engine24 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=\"--diag_suppress=20208\">)

#[[
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
set(CUDAToolkit_ROOT /usr/local/cuda)
find_package(CUDAToolkit REQUIRED)

target_include_directories(Engine24 PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(Engine24 ${CUDAToolkit_LIBRARIES})
message("CUDAToolkit Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")]]
