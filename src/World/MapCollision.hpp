#pragma once

#include "World/Camera.hpp"

#include <cstdint>
#include <vector>

namespace H3D {

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
    // (grounded).
    bool moveAABB(Vec3& position, const Vec3& delta, const Vec3& half) const;

    // Closest hit of a ray against the triangles, or -1. Used for aiming at
    // the map (placing windows on surfaces later).
    float rayCast(const Vec3& origin, const Vec3& dir) const;

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
