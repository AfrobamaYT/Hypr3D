#include "Input/InputController.hpp"

#include <cmath>

namespace H3D {

void InputController::addMotion(double dx, double dy) {
    if (!std::isfinite(dx) || !std::isfinite(dy))
        return;

    m_pendingX += dx;
    m_pendingY += dy;
}

bool InputController::consumeLook(float& yawDelta, float& pitchDelta) {
    if (!std::isfinite(m_pendingX) || !std::isfinite(m_pendingY)) {
        m_pendingX = 0.0;
        m_pendingY = 0.0;
        return false;
    }

    if (m_pendingX == 0.0 && m_pendingY == 0.0)
        return false;

    // Sign convention: the delta handed to addMotion() is the compositor's own
    // logical pointer delta (libinput), where Y grows DOWN. That is the same
    // delta CPointerManager::move() adds to the real cursor position, so the
    // two can never disagree: dy > 0 means the mouse moved down and must pitch
    // the view down, like in every FPS. Driving pitch by +pendingY instead
    // inverted the whole vertical control space: the floor behaved like a
    // ceiling, the crosshair swept the opposite way across window content, and
    // dragged windows travelled the wrong way vertically.
    yawDelta   = static_cast<float>(m_pendingX) * m_sensitivity;
    pitchDelta = static_cast<float>(-m_pendingY) * m_sensitivity;

    m_pendingX = 0.0;
    m_pendingY = 0.0;

    return true;
}

void InputController::setSensitivity(float radiansPerCount) {
    if (!std::isfinite(radiansPerCount) || radiansPerCount <= 0.0f) {
        m_sensitivity = kDefaultRadiansPerCount;
        return;
    }

    m_sensitivity = radiansPerCount;
}

void InputController::reset() {
    m_pendingX = 0.0;
    m_pendingY = 0.0;
}

} // namespace H3D
