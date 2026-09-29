#include "World/Picking.hpp"

#include <algorithm>
#include <cmath>

namespace H3D {

namespace {

static Vec3 inverseRotateX(const Vec3& v, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {
        v.x,
        v.y * c - v.z * s,
        v.y * s + v.z * c,
    };
}

static Vec3 inverseRotateY(const Vec3& v, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {
        v.x * c + v.z * s,
        v.y,
        -v.x * s + v.z * c,
    };
}

static Vec3 toLocal(const Vec3& world, const RayQuad& quad) {
    // Model rotation is RY(yaw) * RX(pitch). Its inverse is
    // RX(-pitch) * RY(-yaw).
    return inverseRotateX(
        inverseRotateY(world, -quad.yaw),
        -quad.pitch
    );
}

} // namespace

std::vector<RayHit> rayPickAllQuads(const Vec3& origin, const Vec3& dir,
                                    const std::vector<RayQuad>& quads) {
    std::vector<RayHit> hits;

    if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z) ||
        !std::isfinite(dir.x) || !std::isfinite(dir.y) || !std::isfinite(dir.z))
        return hits;

    for (const auto& quad : quads) {
        if (!quad.pickable || quad.width <= 0.f || quad.height <= 0.f)
            continue;

        const Vec3 LOCAL_ORIGIN = toLocal(origin - quad.center, quad);
        const Vec3 LOCAL_DIR    = toLocal(dir, quad);

        // The local window plane is always z = 0 and the front faces +Z.
        if (std::fabs(LOCAL_DIR.z) < 0.00001f)
            continue;

        const float T = -LOCAL_ORIGIN.z / LOCAL_DIR.z;

        if (T <= 0.f)
            continue;

        const Vec3 LOCAL_POINT = LOCAL_ORIGIN + LOCAL_DIR * T;

        const float HALF_W = quad.width * 0.5f;
        const float HALF_H = quad.height * 0.5f;

        if (std::fabs(LOCAL_POINT.x) > HALF_W ||
            std::fabs(LOCAL_POINT.y) > HALF_H)
            continue;

        RayHit hit;
        hit.hit      = true;
        hit.id       = quad.id;
        hit.distance = T;
        hit.point    = origin + dir * T;

        hit.u = (LOCAL_POINT.x / quad.width) + 0.5f;
        hit.v = 0.5f - (LOCAL_POINT.y / quad.height);

        hits.push_back(hit);
    }

    std::sort(
        hits.begin(),
        hits.end(),
        [](const RayHit& a, const RayHit& b) { return a.distance < b.distance; });

    return hits;
}

RayHit rayPickQuads(const Vec3& origin, const Vec3& dir,
                    const std::vector<RayQuad>& quads) {
    const auto HITS = rayPickAllQuads(origin, dir, quads);

    if (HITS.empty()) {
        RayHit miss;
        miss.distance = 0.f;
        return miss;
    }

    return HITS.front();
}

} // namespace H3D
