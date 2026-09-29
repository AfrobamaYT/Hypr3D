#pragma once

#include "../World/Camera.hpp"
#include "../World/Picking.hpp"

#include <cstdint>
#include <vector>

namespace H3D {

// Pure 3D world state for window entities: independent positions and sizes,
// center-ray picking, and crosshair drag. No Hyprland dependencies.
namespace World3D {

// How many logical pixels one world unit represents.
constexpr float LOGICAL_PX_PER_UNIT = 100.0f;

// Conversion helpers: logical px -> world units.
inline float toWorld(float logicalPx) {
    return logicalPx / LOGICAL_PX_PER_UNIT;
}

inline float toLogical(float worldUnits) {
    return worldUnits * LOGICAL_PX_PER_UNIT;
}

// A window entity in the world. Quads are axis-aligned, facing +Z
// (towards the initial camera pose).
struct SEntity {
    std::uintptr_t id = 0;

    Vec3  center{0.0f, 0.0f, 0.0f}; // world center
    float width  = 0.0f;            // world units
    float height = 0.0f;            // world units
    float yaw    = 0.0f;             // radians
    float pitch  = 0.0f;             // radians

    // Visual scale applied on top of the box-to-world conversion. New windows
    // spawn fitted into a modest size; the aspect is preserved (the quad must
    // stay proportional to the captured box or the content distorts).
    float spawnScale = 1.0f;

    // Logical window geometry in monitor-local px, fixed at entity creation.
    // Drives the snapshot UV subrect and surface-local click mapping and does
    // NOT change when the entity is dragged around the world.
    // logicalLeft/Top/Width/Height span the FULL decorated box (content +
    // borders); the client surface sits inside it at surfaceOffset with
    // surfaceSize, which is what converts a quad hit to surface-local input.
    float logicalLeft   = 0.0f;
    float logicalTop    = 0.0f;
    float logicalWidth  = 0.0f;
    float logicalHeight = 0.0f;
    float surfaceOffsetX = 0.0f;
    float surfaceOffsetY = 0.0f;
    float surfaceWidth   = 0.0f;
    float surfaceHeight  = 0.0f;
};

// Ray hit against one entity quad. Same shape as the shared ray/quad result:
// u/v are quad-local [0..1], origin at the top-left corner of the window with
// v growing downwards, matching surface-local logical coordinates.
using SHit = RayHit;

// Drag state: keep the grabbed point on the camera centre-ray at exactly the
// depth where the grab began. The window itself smoothly billboards toward the
// camera while it is carried.
struct SDrag {
    bool           active = false;
    std::uintptr_t id     = 0;
    float          distance = 0.0f;
    Vec3           grabLocal{0.0f, 0.0f, 0.0f};

    // Wheel zoom writes the target; the actual distance glides toward it
    // (exponential lerp in updateDrag), so scrolling never teleports.
    double         targetDistance = 0.0;
};

class CWorld {
  public:
    // Replace the entity list (keeps world positions for ids already known
    // when `preservePositions` is set).
    void setEntities(std::vector<SEntity>&& entities, bool preservePositions);

    const std::vector<SEntity>& entities() const {
        return m_entities;
    }

    SEntity* find(std::uintptr_t id);
    void     remove(std::uintptr_t id);
    void     clear();

    // Center-ray hit test against every oriented entity quad.
    // Returns the closest hit. Delegates to the shared ray/quad implementation
    // so world picking and renderer picking can never disagree.
    SHit pick(const Vec3& origin, const Vec3& dir) const;

    // Same test, every hit sorted nearest first -- lets the caller skip
    // transparent pixels of invisible layer overlays.
    std::vector<SHit> pickAll(const Vec3& origin, const Vec3& dir) const;

    // Center of an entity as a hit point (used for the initial drag plane).
    Vec3 centerOf(std::uintptr_t id) const;

    // --- drag ---
    bool startDrag(
        std::uintptr_t id,
        const SHit& hit,
        const Vec3& cameraPosition,
        const Vec3& cameraForward
    );
    void updateDrag(const Vec3& cameraPosition, const Vec3& cameraForward, float dt);
    void stopDrag();

    // Mouse-wheel zoom during a drag: pull the dragged window closer (steps
    // < 0) or push it farther (steps > 0) along the view ray.
    void dragZoom(double steps);

    Vec3 localPoint(std::uintptr_t id, const Vec3& worldPoint) const;
    Vec3 normalOf(std::uintptr_t id) const;
    Vec3 rightOf(std::uintptr_t id) const;
    Vec3 upOf(std::uintptr_t id) const;

    bool      dragActive() const {
        return m_drag.active;
    }
    std::uintptr_t draggedId() const {
        return m_drag.id;
    }

  private:
    SDrag m_drag;

  protected:
    std::vector<SEntity> m_entities;
};

} // namespace World3D

} // namespace H3D
