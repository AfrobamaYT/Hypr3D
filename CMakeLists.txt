cmake_minimum_required(VERSION 3.27)

project(hypr3d LANGUAGES CXX)

# hyprpm and the README run cmake without a build type, which compiled the
# plugin's own code unoptimised (-O0, only Jolt had -O2); the official
# hyprland-plugins build with CXXFLAGS ?= -O2. Loading Khronos' FlightHelmet
# held one frame ~260 ms at -O0 and ~60 ms at -O2 (VM, virgl). A packager's
# own build type (makepkg: None) is kept.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# No STB_GNU_UNIQUE symbols: one of them makes glibc refuse to unload the
# library (it is marked NODELETE), so `hyprctl plugin unload` left it mapped
# and the next `plugin load` ran PLUGIN_INIT again on the old, already shut
# down state -- turning the view on then crashed Hyprland (SIGSEGV, measured
# on Hyprland 0.56.2; 162 such symbols came from Hyprland's headers). For
# every target, Jolt included.
add_compile_options(-fno-gnu-unique)

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

# Keep the viewmodel inside the standalone plugin, including hyprpm builds.
# CMake reads the GLB; regeneration needs neither Python nor an asset tool.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Assets/laser-pistol.glb")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/Assets/laser-pistol.glb" LASER_HEX HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," LASER_BYTES "${LASER_HEX}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/LaserPistol.hpp"
    "#pragma once\n#include <cstdint>\nnamespace H3D { inline constexpr std::uint8_t kLaserPistol[] = {${LASER_BYTES}}; }\n")

# --- Jolt Physics: rigid body simulation for scene objects ------------------
# Included directly via Jolt/Jolt.cmake (the source list + the Jolt target),
# bypassing Jolt's top-level app build. Directory-scope link_libraries keeps
# this working no matter what the plugin target is called.
set(PHYSICS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/Jolt")
set(USE_ASSERTS OFF)
set(INTERPROCEDURAL_OPTIMIZATION OFF) # keep plugin builds fast
set(CROSS_PLATFORM_DETERMINISTIC OFF)
include("${PHYSICS_REPO_ROOT}/Jolt/Jolt.cmake")
# Hide Jolt's symbols inside the plugin .so: without this the archive
# exports thousands of JPH:: symbols, and the plugin load collides with the
# global symbol namespace ("multiply defined" on load).
target_compile_options(Jolt PRIVATE -O2 -fvisibility=hidden -fvisibility-inlines-hidden)
target_include_directories(Jolt PUBLIC "${PHYSICS_REPO_ROOT}")
# Jolt turns JPH_DEBUG on -- and with it every assert, each an int3 that
# takes Hyprland down -- whenever NDEBUG is not set, and Jolt.cmake sets
# NDEBUG for its own Release configurations only. Arch packages build with
# CMAKE_BUILD_TYPE=None: every released plugin had them on, and a degenerate
# GJK step while walking through the Backrooms trapped the owner's session
# (2026-10-08, !ioV.IsNearZero() in EPAPenetrationDepth.h). JPH_NO_DEBUG is
# Jolt's own way out (Build/README.md); only a Debug build keeps them.
target_compile_definitions(Jolt PUBLIC "$<$<NOT:$<CONFIG:Debug>>:JPH_NO_DEBUG>")
link_libraries(Jolt)

# Jolt.cmake forces C++17 in directory scope; ours is C++23.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pkg_check_modules(PIXMAN REQUIRED IMPORTED_TARGET pixman-1)
pkg_check_modules(LIBDRM REQUIRED IMPORTED_TARGET libdrm)
pkg_check_modules(XKBCOMMON REQUIRED IMPORTED_TARGET xkbcommon)
pkg_check_modules(HYPRGRAPHICS REQUIRED IMPORTED_TARGET hyprgraphics)
pkg_check_modules(TURBOJPEG REQUIRED IMPORTED_TARGET libturbojpeg)
pkg_check_modules(PANGOCAIRO REQUIRED IMPORTED_TARGET pangocairo cairo)

add_library(hypr3d SHARED
    src/main.cpp
    src/Render/GLScene.cpp
    src/Render/LaserGun.cpp
    src/Render/Overlay.cpp
    src/Render/MapModel.cpp
    src/Render/PlayerModel.cpp
    src/World/MapCollision.cpp
    src/World/Camera.cpp
    src/World/Outline.cpp
    src/World/Picking.cpp
    src/World/World3D.cpp
    src/Input/InputController.cpp
    src/HyprlandCompat/WindowsCompat.cpp
    src/HyprlandCompat/FocusCompat.cpp
    src/HyprlandCompat/WindowCapture.cpp
    src/HyprlandCompat/PointerHook.cpp
)

target_include_directories(hypr3d
    BEFORE PRIVATE
        "${HYPRLAND_INCLUDE}"
        "${HYPRLAND_ROOT}"
        "${HYPRLAND_SRC}"
        "${HYPRLAND_PROTOCOLS_INCLUDE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(hypr3d PRIVATE
    PkgConfig::HYPRGRAPHICS
    PkgConfig::TURBOJPEG
    PkgConfig::PANGOCAIRO
    PkgConfig::PIXMAN
    PkgConfig::LIBDRM
    PkgConfig::XKBCOMMON
    GLESv2
    EGL
    dl
)

set_target_properties(hypr3d PROPERTIES
    PREFIX ""
    OUTPUT_NAME "hypr3d"
)
