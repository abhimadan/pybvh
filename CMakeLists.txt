# TODO: how do we detect pybind11 is already available?

cmake_minimum_required(VERSION 3.21)
project(pybvh)
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

add_library(pybvh_lib STATIC src/bvh.cpp src/bvh_query.cpp src/closest_point.cpp src/vector.cpp src/ray_intersection.cpp)
target_include_directories(pybvh_lib PUBLIC include)

pybind11_add_module(pybvh pybvh.cpp)
target_link_libraries(pybvh PUBLIC pybvh_lib)
