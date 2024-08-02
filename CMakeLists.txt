cmake_minimum_required(VERSION 3.4.1)
project(rdma_from_gpu)

## Find and setup CUDA at the very beginning
## TODO Use parameters
set(ENV{CUDA_PATH} "/usr/local/cuda")
#set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "75;80;89")
set(CUDA_ARCHITECTURES "75")
#find_package(CUDA)
#find_package(CUDAToolkit)

project(rdma_from_gpu LANGUAGES  C CXX CUDA)
#set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fdiagnostics-color=auto")
#set(CXX "g++-8")
#set(CC "gcc-8")
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set (CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE STREQUAL "" OR (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else ()
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -g -fno-inline")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -g")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    endif ()
endif ()

option(NO_OMIT_FRAME_POINTER "Pass -no-omit-frame-pointer to compiler. You'd like to want that for profiling" OFF)
if (NO_OMIT_FRAME_POINTER)
    message("Not omitting Frame Pointer (a.k.a. using 1 more CPU register in function calls...)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
endif ()
unset(NO_OMIT_FRAME_POINTER CACHE)


option(CCACHE "Accelerate compilation with ccache" ON)
if (CCACHE)
    find_program(CCACHE_FOUND ccache)
    if (CCACHE_FOUND)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        message("Speeding up compilation (hopefully) via ccache")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    endif (CCACHE_FOUND)
endif(CCACHE)
unset(CCACHE CACHE)

option(CUDA_TRACING "Enable NVTX calls to trace and profile with CUDA Nsight" OFF)
if (CUDA_TRACING)
        message("Enable NVTX calls to trace and profile with CUDA Nsight")
        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_TRACING=1")
        #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DCUDA_TRACING=1")
        add_compile_definitions("CUDA_TRACING=1")
endif ()
unset(CUDA_TRACING CACHE)



if ("$ENV{RDMA_CORE_HOME}" STREQUAL "")
    find_path(RDMA_CORE_HOME ../rdma-core/providers/mlx5/mlx5.c HINTS ./rdma-core ../rdma-core NOCACHE)
else ()
    set(RDMA_CORE_HOME $ENV{RDMA_CORE_HOME})
endif ()
message("rdma-core found at ${RDMA_CORE_HOME}")


#set(CUDA_ARCHITECTURES "75;80;89" CACHE STRING "CUDA architectures to pass to NVCC")
set(CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES} CACHE STRING "CUDA architectures to pass to NVCC")

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Build type: " ${CMAKE_BUILD_TYPE})
message("CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message("Using NVCC at ${CMAKE_CUDA_COMPILER}")
message("NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message("Using host compiler at ${CMAKE_CUDA_HOST_COMPILER}")
message("CUDA architectures: ${CUDA_ARCHITECTURES}")
message("Prepending compile commands with ${RULE_LAUNCH_COMPILE}")

find_package(Threads REQUIRED)
find_package(PkgConfig)

set(RDMA_CORE_LIB ${RDMA_CORE_HOME}/build/lib)
set(RDMA_CORE_INCLUDE ${RDMA_CORE_HOME}/build/include)
set(RDMA_CORE_INCLUDE_MLX5 ${RDMA_CORE_HOME}/providers/mlx5/)
set(RDMA_CORE_INCLUDE_SRC ${RDMA_CORE_HOME}/)

message("INCLUDE  ${RDMA_CORE_INCLUDE} ${RDMA_CORE_INCLUDE_SRC} ${RDMA_CORE_INCLUDE_MLX5}")

link_directories(
    ${RDMA_CORE_LIB}
)


# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
# * CMAKE_INSTALL_BINDIR
# * CMAKE_INSTALL_INCLUDEDIR
include(GNUInstallDirs)
# We have sub-projects, but we want CMAKE to put all binaries together
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


add_library(rdma-shim STATIC rdma_shim.c)
set_target_properties(rdma-shim PROPERTIES
    LANGUAGE C)

target_include_directories(rdma-shim PUBLIC
    ${RDMA_CORE_INCLUDE}
    ${RDMA_CORE_INCLUDE_LIB}
    ${RDMA_CORE_INCLUDE_SRC} 
    ${RDMA_CORE_INCLUDE_MLX5})

target_link_libraries(rdma-shim
    #rdma_core
    mlx5
    ibverbs)


add_library(rdma-shim-cuda STATIC rdma_shim.cu)
set_target_properties(rdma-shim-cuda PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )

target_compile_options(rdma-shim-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)

target_include_directories(rdma-shim-cuda PUBLIC
    ${RDMA_CORE_INCLUDE}
    ${RDMA_CORE_INCLUDE_SRC}
    ${RDMA_CORE_INCLUDE_MLX5})

target_link_libraries(rdma-shim-cuda
    mlx5
    ibverbs
    cuda
    rdma-shim
)

# add_library(rdma-shim-cuda-shared SHARED rdma_shim.cu)
# set_target_properties(rdma-shim-cuda-shared PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
#         CUDA_SEPARABLE_COMPILATION ON
#         POSITION_INDEPENDENT_CODE ON
#         )
# target_link_libraries(rdma-shim-cuda-shared
#     mlx5
#     ibverbs
#     cuda
#     rdma-shim)

# target_include_directories(rdma-shim-cuda-shared PUBLIC
#     glog::glog
#     ${RDMA_CORE_INCLUDE}
#     ${RDMA_CORE_INCLUDE_SRC}
#     ${RDMA_CORE_INCLUDE_MLX5}
# )


