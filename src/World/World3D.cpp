#include "World/World3D.hpp"

#include <algorithm>
#include <cmath>

namespace H3D::World3D {

namespace {

static Vec3 rotateX(const Vec3& v, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {
        v.x,
        v.y * c - v.z * s,
        v.y * s + v.z * c,
    };
}

static Vec3 rotateY(const Vec3& v, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {
        v.x * c + v.z * s,
        v.y,
        -v.x * s + v.z * c,
    };
}

static Vec3 rotateZ(const Vec3& v, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {
        v.x * c - v.y * s,
        v.x * s + v.y * c,
        v.z,
    };
}

static Vec3 inverseRotateZ(const Vec3& v, float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {
        v.x * c + v.y * s,
        -v.x * s + v.y * c,
        v.z,
    };
}

// Full window orientation: RY(yaw) * RX(pitch) * RZ(roll). The roll rotates
// the content around the window normal and MUST be part of every axis
// computation, or resize/plane math lands on the unrolled frame.
static Vec3 rotateLocal(const Vec3& v, float yaw, float pitch, float roll) {
    return rotateY(rotateX(rotateZ(v, roll), pitch), yaw);
}

static Vec3 inverseRotate(const Vec3& v, float yaw, float pitch, float roll) {
    return rotateZ(rotateX(rotateY(v, -yaw), -pitch), -roll);
}

static float wrapPi(float a) {
    while (a > 3.14159265358979323846f)
        a -= 6.28318530717958647692f;
    while (a < -3.14159265358979323846f)
        a += 6.28318530717958647692f;
    return a;
}

static float smoothAngle(float current, float target, float dt, float speed) {
    if (!std::isfinite(dt) || dt <= 0.0f)
        return current;
    const float alpha = 1.0f - std::exp(-speed * std::min(dt, 0.1f));
    return current + wrapPi(target - current) * alpha;
}

} // namespace

void CWorld::setEntities(std::vector<SEntity>&& list, bool preservePositions) {
    for (auto& incoming : list) {
        if (preservePositions) {
            for (const auto& existing : m_entities) {
                if (existing.id == incoming.id) {
                    incoming.center = existing.center;
                    break;
                }
            }
        }
    }

    m_entities = std::move(list);
}

SEntity* CWorld::find(std::uintptr_t id) {
    for (auto& entity : m_entities) {
        if (entity.id == id)
            return &entity;
    }

    return nullptr;
}

void CWorld::remove(std::uintptr_t id) {
    std::erase_if(m_entities, [id](const SEntity& e) { return e.id == id; });

    if (m_drag.active && m_drag.id == id)
        m_drag.active = false;
}

void CWorld::clear() {
    m_entities.clear();
    m_drag.active = false;
}

SHit CWorld::pick(const Vec3& origin, const Vec3& dir) const {
    const auto HITS = pickAll(origin, dir);

    if (HITS.empty()) {
        SHit miss;
        miss.distance = 0.f;
        return miss;
    }

    return HITS.front();
}

std::vector<SHit> CWorld::pickAll(const Vec3& origin, const Vec3& dir) const {
    std::vector<RayQuad> QUADS;
    QUADS.reserve(m_entities.size());

    for (const auto& entity : m_entities) {
        RayQuad quad;
        quad.id      = entity.id;
        quad.center  = entity.center;
        quad.width   = entity.width;
        quad.height  = entity.height;
        quad.yaw     = entity.yaw;
        quad.pitch   = entity.pitch;
        quad.roll    = entity.roll;
        quad.pickable = entity.width > 0.0f && entity.height > 0.0f;
        QUADS.emplace_back(quad);
    }

    return rayPickAllQuads(origin, dir, QUADS);
}

Vec3 CWorld::centerOf(std::uintptr_t id) const {
    for (const auto& entity : m_entities) {
        if (entity.id == id)
            return entity.center;
    }

    return {};
}

bool CWorld::startDrag(
    std::uintptr_t id,
    const SHit& hit,
    const Vec3& cameraPosition,
    const Vec3& cameraForward
) {
    (void)cameraForward;

    const SEntity* ENTITY = nullptr;

    for (const auto& entity : m_entities) {
        if (entity.id == id) {
            ENTITY = &entity;
            break;
        }
    }

    if (!ENTITY || !hit.hit)
        return false;

    // Preserve the actual camera-to-grab distance. The dragged window then
    // follows the current centre ray at exactly that range instead of being
    // locked to a camera-relative depth plane.
    const Vec3 OFFSET = hit.point - cameraPosition;
    const float DIST = std::sqrt(dot(OFFSET, OFFSET));

    if (!std::isfinite(DIST) || DIST <= 0.05f)
        return false;

    m_drag.active = true;
    m_drag.id = id;
    m_drag.distance = DIST;
    m_drag.targetDistance = DIST;
    // grabLocal is the grabbed point's offset from the centre in WORLD units
    // (hit.point and center are both world), so it needs no scale factor:
    // updateDrag rotates it back with the same pose and subtracts it from the
    // carried point.
    m_drag.grabLocal = inverseRotate(
        hit.point - ENTITY->center,
        ENTITY->yaw,
        ENTITY->pitch,
        ENTITY->roll
    );

    return true;
}

void CWorld::updateDrag(
    const Vec3& cameraPosition,
    const Vec3& cameraForward,
    float dt
) {
    if (!m_drag.active)
        return;

    SEntity* ENTITY = find(m_drag.id);

    if (!ENTITY) {
        m_drag.active = false;
        return;
    }

    const Vec3 DIR = normalize(cameraForward);
    if (std::fabs(dot(DIR, DIR)) < 0.99f)
        return;

    // Glide the actual distance toward the wheel target -- exponential lerp,
    // so scrolling zooms smoothly instead of teleporting between detents.
    m_drag.distance += (m_drag.targetDistance - m_drag.distance) *
        (1.0 - std::exp(-8.0 * dt));

    // The grabbed point is carried along the camera centre ray at exactly the
    // same camera-to-grab distance recorded when the window was taken.
    const Vec3 TARGET_POINT = cameraPosition + DIR * m_drag.distance;

    // The window's +Z normal should face the camera. Smooth the yaw/pitch so
    // turning the camera never snaps the window. In this model's convention
    // (RY(yaw) * RX(pitch), with Mat4::rotationX mapping (0,0,1) to
    // (0,-sin p, cos p)) the normal's Y component is -sin(pitch), so matching
    // it against -forward yields target pitch = +camera pitch:
    // asin(-TO_CAMERA.y). Using asin(+TO_CAMERA.y) instead flipped the tilt on
    // the vertical axis, leaning the window away from the camera while dragged.
    const Vec3 TO_CAMERA = DIR * -1.0f;
    const float TARGET_YAW = std::atan2(TO_CAMERA.x, TO_CAMERA.z);
    const float TARGET_PITCH =
        std::asin(std::clamp(-TO_CAMERA.y, -1.0f, 1.0f));

    ENTITY->yaw = smoothAngle(ENTITY->yaw, TARGET_YAW, dt, 9.0f);
    ENTITY->pitch = smoothAngle(ENTITY->pitch, TARGET_PITCH, dt, 9.0f);
    ENTITY->pitch = std::clamp(ENTITY->pitch, -1.5f, 1.5f);
    // Full facing: the roll eases to zero as well. A dragged window with a
    // leftover wheel-roll would face the player tilted.
    ENTITY->roll = smoothAngle(ENTITY->roll, 0.0f, dt, 9.0f);

    const Vec3 GRAB_WORLD = rotateLocal(
        m_drag.grabLocal, ENTITY->yaw, ENTITY->pitch, ENTITY->roll
    );

    ENTITY->center = TARGET_POINT - GRAB_WORLD;
}

Vec3 CWorld::localPoint(std::uintptr_t id, const Vec3& worldPoint) const {
    for (const auto& entity : m_entities) {
        if (entity.id == id)
            return inverseRotate(worldPoint - entity.center, entity.yaw,
                                 entity.pitch, entity.roll);
    }
    return {};
}

Vec3 CWorld::normalOf(std::uintptr_t id) const {
    for (const auto& entity : m_entities) {
        if (entity.id == id)
            return rotateLocal({0.0f, 0.0f, 1.0f}, entity.yaw, entity.pitch,
                               entity.roll);
    }
    return {0.0f, 0.0f, 1.0f};
}

Vec3 CWorld::rightOf(std::uintptr_t id) const {
    for (const auto& entity : m_entities) {
        if (entity.id == id)
            return rotateLocal({1.0f, 0.0f, 0.0f}, entity.yaw, entity.pitch,
                               entity.roll);
    }
    return {1.0f, 0.0f, 0.0f};
}

Vec3 CWorld::upOf(std::uintptr_t id) const {
    for (const auto& entity : m_entities) {
        if (entity.id == id)
            return rotateLocal({0.0f, 1.0f, 0.0f}, entity.yaw, entity.pitch,
                               entity.roll);
    }
    return {0.0f, 1.0f, 0.0f};
}

void CWorld::stopDrag() {
    m_drag.active = false;
}

void CWorld::dragZoom(double steps) {
    if (!m_drag.active)
        return;

    // Multiplicative per detent: equal perceived zoom steps at any distance.
    // Writes the TARGET; updateDrag glides the actual distance toward it.
    // No distance limits: the target stays positive, so the dragged window
    // can come arbitrarily close to the camera or recede arbitrarily far.
    constexpr double ZOOM_FACTOR = 1.06;

    m_drag.targetDistance *= std::pow(ZOOM_FACTOR, steps);
}

} // namespace H3D::World3D
