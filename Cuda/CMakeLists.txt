find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

#[[add_library(BcgCudaLib STATIC)
target_link_libraries(BcgCudaLib PRIVATE ${CUDAToolkit_LIBRARIES})]]
#[[
get_target_property(ENGINE24_INCLUDE_DIRS Engine24 INCLUDE_DIRECTORIES)

target_include_directories(BcgCudaLib PUBLIC .)
target_include_directories(BcgCudaLib PUBLIC ${ENGINE24_INCLUDE_DIRS})
]]

target_link_libraries(Engine24 PRIVATE ${CUDAToolkit_LIBRARIES})

add_subdirectory(include)
add_subdirectory(src)

target_sources(Engine24 PRIVATE
        src/hem_test.cu
        src/kmeans.cu
        src/BVHCuda.cu
        src/local_gaussians.cu
)
#target_link_libraries(Engine24 PRIVATE BcgCudaLib)


#[[target_compile_options(BcgCudaLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(BcgCudaLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(BcgCudaLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=\"--diag_suppress=20208\">)]]

target_compile_options(Engine24 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(Engine24 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(Engine24 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=\"--diag_suppress=20208\">)