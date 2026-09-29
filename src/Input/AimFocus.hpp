#pragma once

#include <cstdint>

namespace H3D {

// Decides which window should own keyboard focus, given which window the
// crosshair (screen centre / camera axis) currently points at.
//
// Rules:
//  - aiming at a window focuses it immediately (Minecraft-style target block);
//  - aiming at nothing keeps the previous target for `lostGraceSeconds`, then
//    drops focus entirely. The short grace period stops focus from flickering
//    off and on while the crosshair sweeps across a thin gap or a window edge.
//
// Pure logic, no Hyprland dependencies: id 0 means "no window".
class AimFocus {
  public:
    explicit AimFocus(float lostGraceSeconds = 0.08f) : m_lostGrace(lostGraceSeconds) {}

    void reset() {
        m_current   = 0;
        m_lostTimer = 0.0f;
    }

    // Call once per frame. Returns the window id that should hold focus (0 = none).
    std::uintptr_t update(std::uintptr_t aimedId, float dtSeconds) {
        if (aimedId != 0) {
            m_current   = aimedId;
            m_lostTimer = 0.0f;
            return m_current;
        }

        if (m_current != 0) {
            m_lostTimer += dtSeconds;
            if (m_lostTimer >= m_lostGrace) {
                m_current   = 0;
                m_lostTimer = 0.0f;
            }
        }

        return m_current;
    }

    std::uintptr_t current() const { return m_current; }

  private:
    float          m_lostGrace = 0.08f;
    float          m_lostTimer = 0.0f;
    std::uintptr_t m_current   = 0;
};

} // namespace H3D
