cmake_minimum_required(VERSION 3.27)

project(hypr3d17 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HYPRLAND_HEADERS "" CACHE PATH
    "Hyprland headers root containing include/hyprland"
)

if(NOT HYPRLAND_HEADERS)
    message(FATAL_ERROR
        "Pass -DHYPRLAND_HEADERS=/path/to/headersRoot"
    )
endif()

set(HYPRLAND_INCLUDE
    "${HYPRLAND_HEADERS}/include"
)

set(HYPRLAND_ROOT
    "${HYPRLAND_INCLUDE}/hyprland"
)

set(HYPRLAND_SRC
    "${HYPRLAND_ROOT}/src"
)

set(HYPRLAND_PROTOCOLS_INCLUDE
    "${HYPRLAND_ROOT}/protocols"
)

if(NOT EXISTS
    "${HYPRLAND_INCLUDE}/hyprland/src/plugins/PluginAPI.hpp"
)
    message(FATAL_ERROR
        "PluginAPI.hpp not found:\n"
        "${HYPRLAND_INCLUDE}/hyprland/src/plugins/PluginAPI.hpp"
    )
endif()

if(NOT EXISTS
    "${HYPRLAND_PROTOCOLS_INCLUDE}/color-management-v1.hpp"
)
    message(FATAL_ERROR
        "color-management-v1.hpp not found:\n"
        "${HYPRLAND_PROTOCOLS_INCLUDE}/color-management-v1.hpp"
    )
endif()

if(NOT EXISTS
    "${HYPRLAND_SRC}/render/SyncFDManager.hpp"
)
    message(FATAL_ERROR
        "SyncFDManager.hpp not found:\n"
        "${HYPRLAND_SRC}/render/SyncFDManager.hpp"
    )
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(PIXMAN REQUIRED IMPORTED_TARGET pixman-1)
pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm)
pkg_check_modules(XKBCOMMON REQUIRED IMPORTED_TARGET xkbcommon)
pkg_check_modules(HYPRGRAPHICS REQUIRED IMPORTED_TARGET hyprgraphics)

add_library(hypr3d17 SHARED
    src/main.cpp
    src/Render/GLScene.cpp
    src/World/Camera.cpp
    src/World/Picking.cpp
    src/World/World3D.cpp
    src/Input/InputController.cpp
    src/HyprlandCompat/WindowsCompat.cpp
    src/HyprlandCompat/FocusCompat.cpp
    src/HyprlandCompat/WindowCapture.cpp
    src/HyprlandCompat/PointerHook.cpp
)

target_include_directories(hypr3d17
    BEFORE PRIVATE
        "${HYPRLAND_INCLUDE}"
        "${HYPRLAND_ROOT}"
        "${HYPRLAND_SRC}"
        "${HYPRLAND_PROTOCOLS_INCLUDE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(hypr3d17 PRIVATE
    PkgConfig::HYPRGRAPHICS
    PkgConfig::PIXMAN
    PkgConfig::LIBDRM
    PkgConfig::XKBCOMMON
    GLESv2
    EGL
    dl
)

set_target_properties(hypr3d17 PROPERTIES
    PREFIX ""
    OUTPUT_NAME "hypr3d17"
)