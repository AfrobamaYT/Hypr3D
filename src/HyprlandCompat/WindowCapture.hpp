#pragma once

#include <hyprland/src/plugins/PluginAPI.hpp>
#include <hyprland/src/desktop/DesktopTypes.hpp>
#include <hyprland/src/render/Framebuffer.hpp>

#include <cstdint>
#include <unordered_map>

namespace H3D::Compat {

// Owns compositor-rendered snapshots (makeSnapshotFB) of participating windows
// while 3D mode is active. The framebuffer is monitor-sized; the caller uses
// the window's monitor-local box to pick the UV subrect containing the window.
class CWindowCapture {
  public:
    struct SSnapshot {
        SP<Render::IFramebuffer> fb;
        unsigned int             texID  = 0;

        // Composite texture for a window larger than the monitor: it spans
        // the whole decorated box, so the UV subrect is the full [0..1] range
        // and texID stays 0.
        unsigned int             bigTex = 0;
        int                      width  = 0;
        int                      height = 0;

        // Geometry captured WITH the snapshot. The live window box is written
        // after render.pre (the resize path runs at RENDER_LAST_MOMENT), so
        // UVs taken from the live box trail ahead of the captured pixels and
        // smear the content edges during resizes. The quad, the UV subrect and
        // the picking geometry must all follow THIS box instead.
        //
        // fullBox is the window's real box; sampledBox is where the content
        // actually sits INSIDE the captured texture. They differ when the
        // window extends past the monitor viewport: the snapshot pass cannot
        // capture pixels off-screen, so the workspace render offset is warped
        // for the duration of the pass to pull the whole window into view, and
        // sampledBox records the shifted placement.
        CBox     fullBox       = {};
        CBox     sampledBox    = {};
        Vector2D surfaceOffset = {};
        Vector2D surfaceSize   = {};

        // Logical span of the captured texture. It equals the monitor size
        // unless the window is larger than the monitor and the monitor was
        // temporarily enlarged for the snapshot -- the UV subrect must divide
        // by THIS, not by the live monitor size.
        Vector2D texSpan       = {};

        // Downsampled alpha mask of the captured content (layer surfaces
        // only): an invisible overlay must not swallow the crosshair, so
        // picking falls through pixels whose alpha is ~0 to the surface
        // actually underneath.
        std::vector<unsigned char> alphaMask;
        int  alphaW     = 0;
        int  alphaH     = 0;
        bool alphaValid = false;
        unsigned int maskAge = 0;

        // Identity of the client's last-committed buffer at snapshot time:
        // the cheap "did the content change" signal between frames.
        std::uintptr_t lastBuffer = 0;
    };

    // Renders a fresh snapshot of `window` (uses the focused monitor's output).
    // Returns true and updates the stored snapshot on success. Without `force`
    // the snapshot is skipped when neither the client's committed buffer nor
    // the window's box changed since the last one -- the snapshot pass is the
    // heaviest per-frame cost, and idle windows do not need it.
    bool makeSnapshot(const PHLWINDOW& window, const PHLMONITOR& monitor, bool force = false);

    // Same for a layer-shell surface (panel/bar): monitor-sized capture at
    // the layer's monitor-local box, no decorations, no off-screen pull.
    bool makeSnapshotLayer(const PHLLS& layer, const PHLMONITOR& monitor, bool force = false);

  private:
    // Oversized windows: monitor-sized tiles, composited into one texture.
    bool makeTiledSnapshot(
        const PHLWINDOW&  window,
        const PHLMONITOR& monitor,
        const CBox&       fullBox,
        const Vector2D&   surfOffset,
        const Vector2D&   surfSize,
        std::uintptr_t    id,
        bool              force
    );

    void destroySnapshotGL(SSnapshot& snapshot);

  public:

    // True if this id currently has a live snapshot.
    bool has(std::uintptr_t id) const;

    // The stored snapshot info (fb may still be null if never snapshotted).
    const SSnapshot* get(std::uintptr_t id) const;

    // Drop the snapshot for one window (window closed / removed from 3D).
    void release(std::uintptr_t id);

    // Drop everything (exiting 3D mode).
    void releaseAll();

    // Keep only the ids present in `keep`; release the rest.
    void retainOnly(const std::vector<std::uintptr_t>& keep);

    void shutdownGL();

  private:
    std::unordered_map<std::uintptr_t, SSnapshot> m_snapshots;
};

} // namespace H3D::Compat
