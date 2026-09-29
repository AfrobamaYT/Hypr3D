#pragma once

#include <hyprland/src/plugins/PluginAPI.hpp>

#include <cstdint>

namespace H3D::Compat {

// Stable identity of a window for the 3D world (0 = none).
std::uintptr_t windowId(const PHLWINDOW& window);

// Mapped, visible window with this id, or nullptr.
PHLWINDOW findWindowById(std::uintptr_t id);

PHLWINDOW focusedWindow();
PHLMONITOR focusedMonitor();

// Give keyboard focus to `window` without touching the (hidden) 2D cursor.
void focusWindow(const PHLWINDOW& window);

// Drop keyboard focus entirely: no window is active afterwards.
void clearFocus();

} // namespace H3D::Compat
