cmake_minimum_required(VERSION 3.15)
project(OpenMPDesDecryption)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp")
include_directories(/usr/local/Cellar/libomp/9.0.0/include)
link_directories(/usr/local/Cellar/libomp/9.0.0/lib)

add_executable(OpenMPDesDecryption main.cpp)