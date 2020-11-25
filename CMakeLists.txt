cmake_minimum_required(VERSION 3.17)
project(CUDA_Path_Tracing CXX CUDA)

# set(CMAKE_CUDA_FLAGS "-Wall")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr --device-c")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++17")

########################################
# C++ Setup
########################################
set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

########################################
# CUDA Setup
########################################
find_package(CUDA 10 REQUIRED)


add_executable(CUDA_Path_Tracing
        ./src/main.cpp
        ./src/Global.h
        ./src/Vector.h
        ./src/Light.h
        ./src/AreaLight.h
        ./src/OBJ_Loader.h
        ./src/Ray.h
        ./src/Material.h
        ./src/Bounds3.h
        ./src/Triangle.h
        ./src/MeshTriangle.h
        ./src/Intersection.h
        ./src/BVH.h
        ./src/BVH.cpp
        ./src/Scene.h
        ./src/Scene.cpp
        ./src/Renderer.h
        ./src/Renderer.cu )

set_target_properties(
        CUDA_Path_Tracing
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)