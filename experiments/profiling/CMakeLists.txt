cmake_minimum_required(VERSION 3.16)
project(hnsw_profiling LANGUAGES CXX)

add_executable(hnsw_profile profile.cpp)
target_compile_features(hnsw_profile PRIVATE cxx_std_17)
target_include_directories(hnsw_profile PRIVATE ../..)
find_package(Threads REQUIRED)
target_link_libraries(hnsw_profile PRIVATE Threads::Threads)

# Compile both native algorithms in the same translation unit with the same
# settings. Keep PGM/OpenMP construction serial, like the HNSW insertion loop.
option(PROFILING_NATIVE "Optimize for the build host CPU" ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hnsw_profile PRIVATE -O3)
    if(PROFILING_NATIVE)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
        if(HAS_MARCH_NATIVE)
            target_compile_options(hnsw_profile PRIVATE -march=native)
        endif()
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(statistics_test tests/statistics_test.cpp)
    target_compile_features(statistics_test PRIVATE cxx_std_17)
    target_include_directories(statistics_test PRIVATE ../..)
    target_link_libraries(statistics_test PRIVATE Threads::Threads)
    add_test(NAME statistics COMMAND statistics_test ${CMAKE_CURRENT_BINARY_DIR}/statistics-fixture)
endif()

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME profiling_scripts
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/profile_test.py
                         $<TARGET_FILE:hnsw_profile>)
    endif()
endif()
