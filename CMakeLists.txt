cmake_minimum_required(VERSION 3.0.0)
project(MatrixMulti)

include(CTest)
enable_testing()

# include headers
include_directories(./inc)

# library
link_libraries(pthread)

# source files
aux_source_directory(./src SRCS)
# aux_source_directory(./test TEST)
set(TEST ./test/unit_test.cpp)
set(TEST_ID ./test/independent_test.cpp)

# executable files   
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_executable(MatrixMulti ${SRCS} ${TEST})

# add_executable(IndSpeedTest ${TEST_ID})

# compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -g")
add_compile_options(-O2) 
add_compile_options(-funroll-loops)
 # add_compile_options(-fopenmp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
