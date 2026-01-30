cmake_minimum_required(VERSION 3.15)
project(smartgrid_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    )
else()
    target_compile_options(
        _smartgrid_cpp PRIVATE
        -O3
        -fvisibility=hidden
    )
endif()
