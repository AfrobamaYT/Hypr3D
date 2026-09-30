#include "Input/InputController.hpp"

#include <cmath>

namespace H3D {

void InputController::addMotion(double dx, double dy) {
    if (!std::isfinite(dx) || !std::isfinite(dy))
        return;

    m_pendingX += dx;
    m_pendingY += dy;
}

bool InputController::consumeLook(float& yawDelta, float& pitchDelta, float dt) {
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
    if (dt > 0.0f && m_lookSmoothing > 0.0f) {
        const double FRACTION =
            1.0 - std::exp(-static_cast<double>(dt) / m_lookSmoothing);
        const double TAKE_X = m_pendingX * FRACTION;
        const double TAKE_Y = m_pendingY * FRACTION;

        m_pendingX -= TAKE_X;
        m_pendingY -= TAKE_Y;

        // Below a thousandth of a count the remainder is imperceptible; drain
        // it so the glide always terminates instead of drifting forever.
        if (std::fabs(m_pendingX) < 0.001)
            m_pendingX = 0.0;
        if (std::fabs(m_pendingY) < 0.001)
            m_pendingY = 0.0;

        yawDelta   = static_cast<float>(TAKE_X) * m_sensitivity;
        pitchDelta = static_cast<float>(-TAKE_Y) * m_sensitivity;
    } else {
        yawDelta   = static_cast<float>(m_pendingX) * m_sensitivity;
        pitchDelta = static_cast<float>(-m_pendingY) * m_sensitivity;

        m_pendingX = 0.0;
        m_pendingY = 0.0;
    }

    if (yawDelta == 0.0f && pitchDelta == 0.0f)
        return false;

    return true;
}

void InputController::setLookSmoothing(float seconds) {
    m_lookSmoothing =
        (std::isfinite(seconds) && seconds > 0.0f) ? seconds : 0.0f;
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
