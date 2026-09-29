#pragma once

#include "World/Camera.hpp"

#include <cstdint>
#include <vector>

namespace H3D {

// An axis-aligned window quad facing +Z (towards the initial camera pose).
// This is the one shape the crosshair can hit; both the renderer and the
// world keep their views of it in terms of this struct so that "what you see"
// and "what you can aim at" cannot drift apart.
struct RayQuad {
    std::uintptr_t id = 0;

    Vec3  center{};
    float width  = 0.f;
    float height = 0.f;
    float yaw    = 0.f;
    float pitch  = 0.f;
    float roll   = 0.f;   // around the quad normal; keeps u/v content-aligned

    // False for anything that is drawn but not interactable (uncaptured
    // texture, fully transparent during the 2D->3D fade).
    bool pickable = true;
};

// Intersection result. u/v are quad-local [0..1], origin at the top-left
// corner of the window with v growing downwards, matching surface-local
// logical coordinates that a click is delivered in.
struct RayHit {
    bool           hit = false;
    std::uintptr_t id  = 0;

    Vec3  point{};
    float u        = 0.f;
    float v        = 0.f;
    float distance = 0.f;
};

// Closest hit along the ray. Rays hitting the quad from behind (T <= 0) are
// ignored, as are degenerate/edge-on quads.
RayHit rayPickQuads(const Vec3& origin, const Vec3& dir,
                    const std::vector<RayQuad>& quads);

// Every hit along the ray, sorted nearest first. Lets the caller skip
// transparent pixels (invisible layer overlays) and fall through to the
// surface actually under the crosshair.
std::vector<RayHit> rayPickAllQuads(const Vec3& origin, const Vec3& dir,
                                    const std::vector<RayQuad>& quads);

} // namespace H3D
