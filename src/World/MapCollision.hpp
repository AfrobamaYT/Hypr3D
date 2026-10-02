#pragma once

#include "World/Camera.hpp"

#include <cstdint>
#include <vector>

namespace H3D {

// Two-sided Möller–Trumbore against one triangle; true on hit with the
// distance in outT. Free function so the header's template can use it.
bool mapRayTriangle(const Vec3& o, const Vec3& d, const Vec3& a,
                    const Vec3& b, const Vec3& c, float& outT);

// BVH over the map's world-space triangles, answering the player's per-axis
// collision queries. Pure math: no GL, no Hyprland -- unit-testable.
//
// Resolution is per-axis, minecraft-style: move along X, clamp against every
// triangle the player box overlaps, then Z, then Y (that order makes sliding
// along walls free). This is the same scheme the old window-based
// resolvePlayerCollisions used; only the triangle source changed.
class CMapCollision {
  public:
    struct STL {
        Vec3 a, b, c;

        STL() = default;
        STL(const Vec3& A, const Vec3& B, const Vec3& C) : a(A), b(B), c(C) {}

        // Accepts any triangle type exposing .a/.b/.c (MapModel::STL), so
        // renderer and collision never need to share one struct.
        template <class T>
        STL(const T& t) : a(t.a), b(t.b), c(t.c) {}
    };

    // Rebuilds the tree over `triangles`. Works with any triangle type that
    // exposes .a/.b/.c Vec3 fields (MapModel::STL and CMapCollision::STL),
    // so the renderer and collision never need to share a struct. The
    // triangles are copied: the caller's storage may go away afterwards.
    template <class TriangleRange>
    void build(const TriangleRange& triangles) {
        m_tris.clear();
        m_tris.reserve(triangles.size());
        for (const auto& T : triangles)
            m_tris.push_back({T.a, T.b, T.c});
        buildTree();
    }

    void clear();

    bool empty() const {
        return m_tris.empty();
    }

    size_t triangleCount() const {
        return m_tris.size();
    }

    // Moves `position` by `delta` axis-by-axis (X, Z, Y), clamping against
    // every triangle the player box overlaps. `half` is the player box's
    // half extents. Returns true when downward Y motion was blocked
    // (grounded). Kept for tests; the player walks on moveCapsule().
    bool moveAABB(Vec3& position, const Vec3& delta, const Vec3& half) const;

    // Vertical-capsule mover: `feet` is the body's bottom-centre, the
    // capsule's segment runs feet.y + radius .. feet.y + height - radius.
    // The move is applied whole, then the capsule is depenetrated against
    // nearby triangles (iterative push-out along the shortest separation) --
    // a rounded hull slides along walls and over seams where an AABB
    // corner would snag. A step-up probe (STEP above, settle back) handles
    // thresholds the push cannot climb. Returns grounded (pushed up by a
    // mostly-upward contact).
    // `ceiling` (optional) is set when a contact pushed the capsule DOWN --
    // a head bump; the caller stops any upward jump velocity on it, or the
    // player sticks to the ceiling while gravity slowly bleeds the jump off.
    bool moveCapsule(Vec3& feet, const Vec3& delta, float radius,
                     float height, bool* ceiling = nullptr) const;

    // Per-axis AABB mover against a LIST of trees: the center is moved by
    // delta once, then clamped against every tree per axis (X, Z, Y).
    // groundedOut = downward Y clamp happened (the box landed). This is the
    // moving-object proxy step for scene-object physics.
    static void moveAABBOn(const std::vector<const CMapCollision*>& trees,
                           Vec3& center, const Vec3& delta, const Vec3& half,
                           bool& groundedOut);

    // The same mover, but against a LIST of trees (the per-object collision
    // architecture: one tree per scene object + the grid platform -- a
    // released object rebuilds only ITS OWN small tree, the static map's
    // big tree never rebuilds for it).
    static bool moveCapsuleOn(const std::vector<const CMapCollision*>& trees,
                              Vec3& feet, const Vec3& delta, float radius,
                              float height, bool* ceiling = nullptr);

    // Closest hit of a ray against the triangles, or -1. Used for aiming at
    // the map (placing windows on surfaces later).
    float rayCast(const Vec3& origin, const Vec3& dir) const;

    // Collect triangle indices whose AABBs intersect the box (the internal
    // BVH query) -- the multi-tree mover gathers candidates this way.
    void collect(const Vec3& min, const Vec3& max,
                 std::vector<uint32_t>& out) const {
        query(min, max, out);
    }

    // Brute-force closest hit against an arbitrary triangle set (two-sided
    // Möller–Trumbore). Shared with per-model picking; accepts any triangle
    // type exposing .a/.b/.c.
    template <class TriangleRange>
    static float rayTriangles(const TriangleRange& triangles,
                              const Vec3& origin, const Vec3& dir) {
        float best = -1.f;
        for (const auto& T : triangles) {
            float t = -1.f;
            if (mapRayTriangle(origin, dir, T.a, T.b, T.c, t) &&
                (best < 0.f || t < best))
                best = t;
        }
        return best;
    }

  private:
    struct SNode {
        Vec3     min{}, max{};
        uint32_t start = 0, count = 0; // triangle range for leaves
        int32_t  left = -1, right = -1;
    };

    static bool triBoxOverlap(const STL& t, const Vec3& boxCenter,
                              const Vec3& boxHalf);

    // Median-split BVH over m_tris (uses the copied STL triangles). Leaves
    // index into m_order, the permutation that sorts triangles by subtree.
    void buildTree();

    void query(const Vec3& min, const Vec3& max,
               std::vector<uint32_t>& out) const;

    std::vector<STL>          m_tris;
    std::vector<uint32_t>     m_order; // leaf range -> triangle index
    std::vector<SNode>        m_nodes;
};

} // namespace H3D
