cmake_minimum_required(VERSION 3.0.0)
project(MatrixMulti)

include(CTest)
enable_testing()

# include headers
include_directories(./inc)

# source files
aux_source_directory(./src SRCS)
set(TEST 
    ./test/unit_test.cpp)

# executable files
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_executable(MatrixMulti ${SRCS} ${TEST})

add_definitions(-std=c++11)
add_definitions(-lpthread)
add_definitions(-Wall -Wextra)
add_definitions(-O2)
add_definitions(-funroll-loops)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
