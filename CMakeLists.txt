cmake_minimum_required(VERSION 3.15)
project(smartgrid_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT CMAKE_BUILD_TYPE AND NOT MSVC)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

set(SMARTGRID_CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/smartgrid/hot_math/bindings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/smartgrid/hot_math/src/smartgrid_predictor.cpp
)

pybind11_add_module(
    _smartgrid_cpp
    ${SMARTGRID_CPP_SOURCES}
)

target_include_directories(
    _smartgrid_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/smartgrid/hot_math/header
)

if (MSVC)
    # Enable optimization ONLY for Release-like builds
    target_compile_options(
        _smartgrid_cpp PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2>
        $<$<CONFIG:Release>:/fp:fast>       
    )
else()
    target_compile_options(
        _smartgrid_cpp
        PRIVATE
        -O3                     # explicit highest optimization
        -ffast-math             # safe for SmartGrid kernel
        -fvisibility=hidden     # reduce symbol exports
    )
endif()

