#pragma once

namespace H3D {

// Accumulates raw relative pointer motion (in pointer counts) and converts it
// into camera yaw/pitch deltas in radians.
//
// Motion is accumulated in double and only converted on consumeLook(), so the
// resulting camera rotation does not depend on how the compositor happens to
// chunk motion events into frames. Rounding per event would eat sub-count
// motion entirely; the old "delta from screen centre + warp" scheme additionally
// lost everything the compositor floored away.
class InputController {
  public:
    static constexpr float kDefaultRadiansPerCount = 0.0025f;

    // Non-finite components are dropped so a single bad event can never poison
    // the accumulator with NaN.
    void addMotion(double dx, double dy);

    // Returns false when there is nothing to apply.
    // yaw is signed for "positive = turn right". Motion arrives in compositor
    // pointer units with Y growing DOWN (libinput convention -- the same delta
    // that moves the real cursor), so pitch is driven by -pendingY: pushing
    // the mouse up looks up, like in every FPS.
    //
    // dt > 0 with look smoothing enabled hands out only a per-frame fraction
    // of the accumulated motion (exponential in dt), so a stopped mouse coasts
    // to a halt instead of cutting dead. dt == 0 drains everything at once.
    bool consumeLook(float& yawDelta, float& pitchDelta, float dt = 0.0f);

    // Glide time constant in seconds after input stops. 0 disables.
    void setLookSmoothing(float seconds);

    // Non-positive / non-finite values fall back to the default.
    void setSensitivity(float radiansPerCount);
    float sensitivity() const { return m_sensitivity; }

    void reset();

  private:
    double m_pendingX    = 0.0;
    double m_pendingY    = 0.0;
    float  m_sensitivity = kDefaultRadiansPerCount;
    float  m_lookSmoothing = 0.0f;
};

} // namespace H3D
