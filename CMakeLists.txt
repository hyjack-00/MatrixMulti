cmake_minimum_required(VERSION 3.0.0)
project(MatrixMulti)

include(CTest)
enable_testing()

# source files
aux_source_directory(./src SRCS)
set(TEST ./test/unit_test.cpp)
set(TEST_ID ./test/independent_test.cpp)
# set(TEST_Ye ./test/ye.cpp)

# targets 
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_executable(MatrixMulti ${SRCS} ${TEST})

add_executable(IndTest ${TEST_ID})


# include headers
target_include_directories(MatrixMulti PUBLIC ./inc)

# library
target_link_libraries(MatrixMulti pthread)
target_link_libraries(IndTest pthread)

# compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -g")
add_compile_options(-O2)
add_compile_options(-fopenmp)
# add_compile_options(-static)
# add_compile_options(-funroll-loops)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
