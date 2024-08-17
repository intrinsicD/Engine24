find_package(CUDAToolkit REQUIRED)

target_include_directories(Engine24 PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(Engine24 PUBLIC .)

target_sources(Engine24 PRIVATE
        PluginCuda.cpp
        CudaCommon.cu
        vec2.cu
        vec3.cu
        vec4.cu
        mat2.cu
        mat3.cu
        mat4.cu
)

target_link_libraries(Engine24 PRIVATE ${CUDAToolkit_LIBRARIES})

set(CMAKE_CUDA_ARCHITECTURES native)

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
target_compile_options(Engine24 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(Engine24 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(Engine24 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=\"--diag_suppress=20208\">)

add_subdirectory(lbvh)
add_subdirectory(Kmeans)
add_subdirectory(LocalGaussians)