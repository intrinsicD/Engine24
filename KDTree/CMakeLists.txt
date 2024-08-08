target_include_directories(Engine24 PUBLIC .)
target_sources(Engine24 PRIVATE
        KDTreeCompute.cpp
        KDTreeCpu.cpp
        KDTreeCuda.cu
)