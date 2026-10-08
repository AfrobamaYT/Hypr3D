#pragma once

#include "Render/MapModel.hpp"
#include "Render/Overlay.hpp"
#include "Render/PlayerModel.hpp"
#include "World/Camera.hpp"
#include "World/Outline.hpp"
#include "World/Picking.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace H3D {

// The vertical field of view of the 3D view, in degrees. Shared with
// main.cpp (fullscreen transition computes the screen-filling quad from it).
static constexpr float kFovDeg = 65.0f;

// The 3D view: a first-person room containing the windows of the active
// workspace. It deliberately knows nothing about Hyprland -- main.cpp feeds it
// WindowRender entries and reads back what the crosshair hit.
class GLScene {
  public:
    // One window as it should appear in the 3D world. Carries exactly what a
    // draw and a pick both need, so the renderer cannot reach into compositor
    // state and the two can never disagree about what is on screen.
    struct WindowRender {
        std::uintptr_t id = 0;

        // 0 = not captured yet. Such a window is neither drawn nor pickable,
        // which is what keeps aim focus from handing keyboard input to
        // something invisible.
        unsigned int texture = 0;

        float x = 0.f, y = 0.f, z = 0.f; // world centre
        float width = 0.f, height = 0.f; // world size, independent per window
        float yaw = 0.f, pitch = 0.f;   // world orientation, radians
        float roll = 0.f;                // in-plane rotation around the normal

        // Subrect of `texture` to sample: (u0,v0) at the quad's bottom-left in
        // GL convention, (u1,v1) at its top-right. Defaults cover the whole
        // texture so a caller with a tightly-fit texture needs no bookkeeping.
        float u0 = 0.f, v0 = 0.f, u1 = 1.f, v1 = 1.f;

        float alpha = 1.0f;

        // Depth slab: thickness in world units, extruded BACKWARDS along the
        // window's normal (0 = the plain flat quad). When `outlines` carries
        // the captured alpha's silhouette, the slab's walls hug that shape --
        // rounded corners stay rounded -- and each wall samples its own
        // silhouette texel, so the window texture's edge colors paint the
        // sides. Without an outline the slab is a plain box.
        float depth = 0.0f;
        std::shared_ptr<const std::vector<SOutlineLoop>> outlines;

        // Crumpled like a sheet of paper: 0 = flat, 1 = a paper ball about
        // a third of the window's size, turned `spin` radians about its up
        // axis. Drawn as a creased mesh instead of the quad (drawCrumpled).
        float crumple = 0.0f;
        float spin    = 0.0f;
    };

    // Where one monitor looks through the view plane, in tangents of the view
    // axis (the plane at distance 1): left, right, bottom, top. The monitor
    // the view is built for gets the symmetric field of view; a neighbour
    // gets the rectangle beside it, so several monitors show one continuous
    // room -- the way CAVEs and triple-screen simulators split a view (Kooima,
    // "Generalized Perspective Projection").
    struct ViewWindow {
        float left = 0.f, right = 0.f, bottom = 0.f, top = 0.f;
    };

    GLScene();

    // Draws into `targetFBO` and composites the result over whatever is
    // already there with `alpha`, so the 2D desktop underneath fades out
    // instead of being captured and re-projected -- the workspace itself is
    // never rendered as an object in the scene.
    //
    // `view` null: the symmetric field of view of this framebuffer. `primary`:
    // the monitor the view is built for -- it advances the clock and the
    // player's animation, loads what the scene needs and draws the HUD and the
    // crosshair; any other monitor only draws the same room through `view`.
    bool render(unsigned int targetFBO, int width, int height, float alpha,
                float dt, const std::vector<WindowRender>& windows,
                const ViewWindow* view = nullptr, bool primary = true);

    // Raw pointer counts -> camera look. Prefer driving this through
    // InputController so sensitivity policy stays in one testable place.
    void mouseMove(float dx, float dy);

    // Already-converted rotation in radians.
    void rotateView(float yawDelta, float pitchDelta);

    // Background panorama: set a path ("" restores the default void). The
    // image is (re)loaded lazily inside render() when the file changes.
    // Equirectangular mapping around the camera.
    void setPanoramaPath(const std::string& path);

    // What the crosshair currently points at, or 0 for "nothing". Ray is cast
    // through the centre of the screen from the camera, never from the OS
    // cursor, so aiming is independent of where the pointer happens to sit.
    std::uintptr_t pick(const std::vector<WindowRender>& windows) const;

    Camera& camera() {
        return m_camera;
    }
    const Camera& camera() const {
        return m_camera;
    }

    void reset();
    void shutdown();

    // The pointer of a window being used (main's g_use): the world point
    // that overlay sprites anchored at EAnchor::Cursor are drawn at.
    void setCursorPoint(const Vec3& at, bool on) {
        m_cursorAt = at, m_cursorOn = on;
    }

    // The crosshair overlay (main's, Overlay::): the mark, the label under
    // it, the gun's pill, a window in use's pointer, as screen sprites.
    enum class EAnchor : uint8_t {
        Center, // the screen's centre: the crosshair
        Top,    // the top middle
        TopLeft,
        Bottom, // the bottom middle
        Cursor, // the pointer of a window in use (setCursorPoint)
        World,  // the world point `at`, through the primary view
    };
    struct SOverlaySprite {
        std::shared_ptr<const Overlay::SImage> image;
        EAnchor anchor = EAnchor::Center;
        float   dy     = 0.0f; // px down from the anchor
        float   alpha  = 1.0f;
        float   dx     = 0.0f; // px right of the anchor
        Vec3    at{};          // EAnchor::World
    };
    void setOverlay(std::vector<SOverlaySprite> sprites) {
        m_overlay = std::move(sprites);
    }

    // F3's room check (main's overlay): how many map objects are loaded,
    // of how many, with how many triangles.
    struct SMapStats {
        size_t loaded = 0, total = 0, triangles = 0;
    };
    SMapStats mapStats() const {
        SMapStats S{0, m_slots.size(), 0};
        for (const auto& SLOT : m_slots)
            if (SLOT.model && SLOT.model->loaded())
                ++S.loaded, S.triangles += SLOT.model->triangles().size();
        return S;
    }

    // The base grid platform (world zero): visible + collidable.
    // F3 debug: the player's collision capsule outline (world space).
    void setPlayerDebugCapsule(const Vec3& center, bool on) {
        m_pDbgCenter = center;
        m_pDbgOn     = on;
    }

    // The companion (hl.plugin.hypr3d.companion): its capsule outline and a
    // line where it looks, hidden by what stands in front of it.
    void setCompanion(const Vec3& center, float yaw, bool on) {
        m_compCenter = center;
        m_compYaw    = yaw;
        m_compOn     = on;
    }

    // The companion's eye (companion("see")): one picture of the room from
    // `eye`, 90 degrees across, drawn with the next frame into a target of
    // its own. It comes back without stalling the compositor: glReadPixels
    // into a pixel pack buffer, mapped once its fence has passed, a frame or
    // more later -- the way OBS's gl-stagesurf.c reads its frames. In it the
    // player stands as his model, or as an orange capsule without one; the
    // companion's own capsule is not drawn.
    static constexpr int kSightWidth = 768, kSightHeight = 432;
    void requestSight(const Camera& eye) {
        m_sightCam    = eye;
        m_sightWanted = true;
    }
    // Forgets a picture asked for or under way.
    void dropSight() {
        m_sightWanted = false;
        m_sightDrop   = m_sightFence != nullptr;
        m_sightResult = ESight::None;
        m_sightPixels.clear();
    }
    enum class ESight { None, Picture, Failed };
    // What the eye finished since the last call: the picture's RGBA rows,
    // bottom up, moved into `rgba`, or why there is none.
    ESight takeSight(std::vector<unsigned char>& rgba, std::string& why) {
        const ESight R = m_sightResult;
        m_sightResult  = ESight::None;
        if (R == ESight::Picture)
            rgba.swap(m_sightPixels);
        if (R == ESight::Failed)
            why = m_sightWhy;
        return R;
    }

    // Portals (hl.plugin.hypr3d.portal): a picture standing on the floor,
    // `width` metres wide and as tall as the picture, facing `yaw` (the
    // camera's convention: it looks along {sin yaw, 0, -cos yaw}). Its
    // transparent pixels are cut out. Pictures load inside render, where
    // the GL context is current.
    struct SPortalSpec {
        std::string name, image;
        Vec3        base{}; // the floor point under its middle
        float       yaw   = 0.0f;
        float       width = 1.8f;
    };
    void setPortals(const std::vector<SPortalSpec>& portals);
    // Height over width of a portal's picture; 0 while it is not loaded.
    float portalAspect(const std::string& name) const;

    // Panels over the picture (main's HUD: the rice's bar and notifications
    // in view, as in 2D), set per monitor before its render. In the target's
    // NDC -- Hyprland's framebuffers put y = -1 at the TOP -- and in UVs of
    // the top-down snapshot texture.
    struct SHudQuad {
        unsigned int texture = 0;
        float        x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f; // left/top, right/bottom
        float        u0 = 0.f, v0 = 0.f, u1 = 0.f, v1 = 0.f; // the same corners
    };
    void setHud(std::vector<SHudQuad>&& quads) { m_hud = std::move(quads); }

    // The desktop's wallpaper over the room and under the windows, on the
    // way in and out (main's flight): the 2D desktop the windows lift off
    // from, fading as the room comes up. In the HUD's coordinates.
    void setBackdrop(const SHudQuad& quad, float alpha) {
        m_backdrop      = quad;
        m_backdropAlpha = alpha;
    }

    // Shadows on the floor under the windows: a soft blob each, as games
    // ground a floating object (a blob shadow) and visionOS a window. main
    // finds the floor and sizes them; centre on the floor, `right` the
    // horizontal unit axis of the window's width.
    struct SShadow {
        Vec3  center{}, right{1.f, 0.f, 0.f};
        float halfW = 0.f, halfD = 0.f, alpha = 0.f;
    };
    void setShadows(std::vector<SShadow>&& shadows) { m_shadows = std::move(shadows); }

    // The window being read or shown big (F2, F4): drawn after the rest of
    // the room and never hidden by it, over the room darkened by `dim`
    // (0 = as it is, 1 = black). id 0: none.
    float featuredDim() const { return m_featuredDim; }
    float featuredBlur() const { return m_featuredBlur; }
    // `blur` (0..1): the room behind it blurred as well, the F1 menu's.
    void setFeatured(std::uintptr_t id, float dim, float blur = 0.0f) {
        m_featuredId   = id;
        m_featuredDim  = dim;
        m_featuredBlur = blur;
    }

    // View zoom (the C key): magnification narrows the render fov
    // symmetrically around the crosshair, so aiming stays exact.
    void setZoom(float magnification) {
        m_zoom = magnification > 0.01f ? magnification : 0.01f;
    }

    // The player's own character (player.mesh). The SAME mesh description
    // the scene objects use: path, transform, material overrides. Config
    // and pose come from main; the animation clock runs inside render().
    struct SPlayerCfg {
        std::string path;
        Vec3        posOffset{};   // model anchor offset from the feet
        Vec3        rotDeg{};      // authored facing correction, XYZ degrees
        Vec3        scale{1.f, 1.f, 1.f};
        float       emissiveScale = 1.0f;
        bool        flat = false;  // true: raw texture, no headlight shading
        std::string center;       // parsed for format parity; the player
        Vec3        centerOffset{}; // rotates around its anchor, so only
                                  // center_offset (the anchor shift) applies
        // idle, walk, run, jump: animation index or name (name wins),
        // plus the playback speed multiplier (1 = as authored).
        int         animIdx[CPlayerModel::kStateCount] = {-1, -1, -1, -1};
        std::string animName[CPlayerModel::kStateCount];
        float       animSpeed[CPlayerModel::kStateCount] = {1.f, 1.f, 1.f, 1.f};
    };

    void setPlayerConfig(const SPlayerCfg& cfg) {
        m_playerCfg = cfg;
        m_playerPath.clear(); // forces a (re)load attempt
    }

    void setPlayerPose(const Vec3& feet, float yawRad) {
        m_playerFeet = feet;
        m_playerYaw  = yawRad;
    }

    void setPlayerVisible(bool on) {
        m_playerVisible = on;
    }

    CPlayerModel* player() {
        return &m_player;
    }

    // The waste bin's glow (Larch's motion draft): a soft ring of light
    // around its rim -- it flashes as a ball goes in and breathes while the
    // bin waits for the application. 0 = none.
    void setBinHalo(const Vec3& rim, float radius, float intensity) {
        m_binHaloAt = rim;
        m_binHaloR  = radius;
        m_binHaloI  = intensity;
    }

    void setGridVisible(bool on) {
        m_gridVisible = on;
    }

    // --- scene: an arbitrary number of glTF objects -------------------------

    // One object's render-relevant config (index-aligned with the slots).
    struct SSceneSpec {
        std::string path;
        Vec3 position{}, rotationDeg{}, scale{1.0f, 1.0f, 1.0f};
        float emissiveScale = 1.0f;
        bool flat = false; // false = headlight half-lambert shading
        CMapModel::ECenter center = CMapModel::ECenter::Logical;
        Vec3 centerOffset{};
    };

    // Diff-apply the object list: new slots are created (files load lazily
    // in refreshScene, EGL current there), changed specs update transforms
    // in place, removed slots free their GL objects. The scene fingerprint
    // folds every model's generation, so collision can rebuild when any
    // object (re)loads or moves.
    void setSceneObjects(const std::vector<SSceneSpec>& specs);

    // Per-model access for collision and the grab interaction.
    size_t sceneModelCount() const {
        return m_slots.size();
    }

    CMapModel* sceneModel(size_t index) {
        return index < m_slots.size() ? m_slots[index].model.get() : nullptr;
    }

    // A scene object is still being read and decoded, so its collision is
    // not there yet.
    bool scenePending() const {
        for (const auto& S : m_slots)
            if (S.model && S.model->pending())
                return true;
        return false;
    }

    // Move/rotate a (grabbed) object: updates the slot spec and the model
    // transform in one place so mtime reloads keep the new placement.
    void setSceneObjectTransform(size_t index, const Vec3& position,
                                 const Vec3& rotationDeg);

    uint64_t sceneFingerprint() const;

    void setMapDebugCollisions(bool on) {
        for (auto& S : m_slots)
            S.model->setDebugCollisions(on);
    }

    // Diagnostics: requests a single pixel readback from the offscreen scene
    // on the next frame. If it comes back with alpha below 255 the composite
    // is see-through by construction, not because of the entry fade.
    void requestProbe();
    bool probeValid() const;
    const unsigned char* probeRGBA() const;

  private:
    bool initialize();
    bool createPrograms();
    bool createMeshes();
    bool ensureSceneFramebuffer(int width, int height);

    void refreshScene();
    void refreshPlayer();


    void destroyGLObjects();

    // texture == 0 draws `color` instead of sampling. uvRect is
    // {u0, v0, u1, v1} in texture space.
    void drawQuad(
        unsigned int vao,
        int vertexCount,
        const Mat4& mvp,
        unsigned int texture,
        const float uvRect[4],
        float r, float g, float b, float a
    );

    void drawFloor(const Mat4& vp);
    void drawGrid(const Mat4& vp);
    void drawPanorama(const ViewWindow& view);
    // Everything in the room through `vp`, into the bound target: the
    // screen's view, or (`sight`) the companion's eye -- no debug lines, no
    // companion, the player always.
    void drawRoom(const Mat4& vp, const ViewWindow& view,
                  const std::vector<WindowRender>& windows, float dt,
                  bool primary, bool sight);
    void refreshPanorama();

    void drawWindows(const Mat4& vp, const std::vector<WindowRender>& windows);
    // A window crumpled like paper (WindowRender::crumple): a creased mesh,
    // opaque and depth-written, drawn before the translucent windows.
    void drawCrumpled(const Mat4& vp, const WindowRender& window);
    // Portals.
    struct SPortalGL {
        SPortalSpec  spec;
        unsigned int tex    = 0;
        float        aspect = 0.0f;
        std::string  loaded; // the image path the texture holds
    };
    std::vector<SPortalGL>    m_portals;
    std::vector<unsigned int> m_portalTrash; // textures to delete in render
    unsigned int              m_portalVAO = 0, m_portalVBO = 0;
    int                       m_sceneAlphaCut = -1;
    void refreshPortals();
    void drawPortals(const Mat4& vp);
    // The HUD (setHud).
    std::vector<SHudQuad> m_hud;
    unsigned int          m_hudVAO = 0, m_hudVBO = 0;
    void drawHud(float alpha);
    // Shadows (setShadows).
    std::vector<SShadow> m_shadows;
    unsigned int         m_shadowTex = 0, m_shadowVAO = 0, m_shadowVBO = 0;
    void drawShadows(const Mat4& vp);
    void drawBinHalo(const Mat4& vp);
    void ensureCrumpleBuffers();
    // The featured window (setFeatured) and the dark over the rest.
    std::uintptr_t m_featuredId  = 0;
    float          m_featuredDim = 0.0f;
    bool           m_windowsOnTop = false; // drawWindows without the depth test
    unsigned int   m_dimVAO = 0, m_dimVBO = 0;
    void drawDim(float dim);
    SHudQuad       m_backdrop{};
    float          m_backdropAlpha = 0.0f;
    void           drawBackdrop();
    // The room blurred behind a featured window (dual Kawase: three halving
    // passes down, three back up), mixed over the sharp room by `amount`.
    struct SBlurLevel {
        unsigned int fbo = 0, tex = 0;
        int          w = 0, h = 0;
    };
    SBlurLevel     m_blurLevels[3];
    unsigned int   m_blurDownProgram = 0, m_blurUpProgram = 0;
    float          m_featuredBlur = 0.0f;
    void           drawBlur(float amount);
    void           releaseBlur();
    void drawFullscreen(float alpha);
    void drawOverlay(int width, int height);

  private:
    bool m_initialized = false;

    int m_sceneWidth = 0;
    int m_sceneHeight = 0;

    unsigned int m_sceneFBO = 0;
    unsigned int m_sceneColor = 0;
    unsigned int m_sceneDepth = 0;

    // Offscreen scenes of the other sizes in use: monitors of different
    // sizes share the room, and reallocating one target for each of them
    // every frame would be the cost of the whole pass.
    struct SceneTarget {
        int          width = 0, height = 0;
        unsigned int fbo = 0, color = 0, depth = 0;
    };
    std::vector<SceneTarget> m_otherSceneTargets;

    unsigned int m_sceneProgram = 0;
    unsigned int m_blitProgram = 0;
    unsigned int m_panoramaProgram = 0;

    // Unit quad in the XY plane, centred on the origin, facing +Z. Windows.
    unsigned int m_quadVAO = 0;
    unsigned int m_quadVBO = 0;
    int          m_quadVertexCount = 0;

    // Dynamic mesh for BSP-ordered window pieces (pos3+uv2, like the scene
    // program's layout).
    unsigned int m_polyVAO = 0;
    unsigned int m_polyVBO = 0;
    unsigned int m_crumpleVAO = 0;
    unsigned int m_crumpleVBO = 0;
    Vec3         m_binHaloAt{};
    float        m_binHaloR = 0.0f, m_binHaloI = 0.0f;

    // Unit quad in the XZ plane. The ground.
    unsigned int m_floorVAO = 0;
    unsigned int m_floorVBO = 0;
    int          m_floorVertexCount = 0;

    unsigned int m_gridVAO = 0;
    unsigned int m_gridVBO = 0;
    int          m_gridVertexCount = 0;

    unsigned int m_fullscreenVAO = 0;
    unsigned int m_fullscreenVBO = 0;

    bool         m_cursorOn = false;
    Vec3         m_cursorAt{};
    Mat4         m_lastVP{}; // the primary view's, for the cursor
    std::vector<SOverlaySprite>                  m_overlay;
    std::unordered_map<uint64_t, unsigned int>   m_overlayTex; // by SImage::serial
    unsigned int                                 m_overlayVAO = 0, m_overlayVBO = 0;

    int m_sceneMVP = -1;
    int m_sceneTexture = -1;
    int m_sceneTextured = -1;
    int m_sceneColorUniform = -1;
    int m_sceneUVRect = -1;
    int m_sceneLodBias = -1;

    int m_blitTexture = -1;
    int m_blitAlpha = -1;

    int m_panoramaFwd = -1;
    int m_panoramaRight = -1;
    int m_panoramaUp = -1;
    int m_panoramaTanX = -1;
    int m_panoramaTanY = -1;
    int m_panoramaTanCenter = -1;
    int m_panoramaLod = -1;
    int m_panoramaSampler = -1;

    std::string m_panoramaPath;   // requested (resolved) path
    std::string m_panoramaLoaded; // path actually loaded into m_panoramaTex
    std::filesystem::file_time_type m_panoramaMtime{};
    bool m_panoramaMtimeValid = false;
    unsigned int m_panoramaTex = 0;
    int m_panoramaW = 0;

    int  m_width = 0, m_height = 0;

    // Single source of truth for the view pose: mouse-look, movement and the
    // crosshair pick all read these same yaw/pitch values.
    Camera m_camera;

    float m_time = 0.0f;

    // F3 debug HUD state.
    bool                            m_gridVisible  = true;
    float                           m_zoom         = 1.0f;
    CPlayerModel                    m_player;
    // Player debug capsule (F3) and the companion, drawn the same way.
    unsigned int                    m_pDbgProgram = 0, m_pDbgVAO = 0, m_pDbgVBO = 0;
    int                             m_pDbgMVP = -1, m_pDbgColor = -1;
    int                             m_pDbgVerts = 0;
    bool                            m_pDbgOn = false;
    Vec3                            m_pDbgCenter{};
    bool                            m_compOn = false;
    Vec3                            m_compCenter{};
    float                           m_compYaw = 0.0f;
    void drawCapsule(const Mat4& vp, const Vec3& center, const float* yaw,
                     const Vec3& color, bool depthTest);
    // The companion's eye, see requestSight.
    Camera                          m_sightCam;
    bool                            m_sightWanted = false;
    bool                            m_sightDrop   = false; // the readback under way is unwanted
    void*                           m_sightFence  = nullptr; // GLsync
    unsigned int                    m_sightFBO = 0, m_sightColor = 0, m_sightDepth = 0;
    unsigned int                    m_sightPBO = 0;
    ESight                          m_sightResult = ESight::None;
    std::vector<unsigned char>      m_sightPixels;
    std::string                     m_sightWhy;
    void drawSight(const std::vector<WindowRender>& windows);
    void readSight();
    void sightFailed(std::string why) {
        m_sightResult = ESight::Failed;
        m_sightWhy    = std::move(why);
    }
    SPlayerCfg                      m_playerCfg;
    std::string                     m_playerPath;      // tilde-expanded
    std::filesystem::file_time_type m_playerMtime{};
    bool                            m_playerMtimeValid = false;
    bool                            m_playerVisible    = false;
    Vec3                            m_playerFeet{};
    float                           m_playerYaw        = 0.f;


    // Scene slots: one per config object, index-aligned with the specs.
    struct SSlot {
        std::unique_ptr<CMapModel>      model;
        SSceneSpec                      spec{};
        std::string                     loadedPath; // expanded
        std::filesystem::file_time_type mtime{};
        bool                            mtimeValid = false;
    };
    std::vector<SSlot>              m_slots;

    bool          m_probeRequested = false;
    bool          m_probeValid     = false;
    unsigned char m_probe[4]       = {0, 0, 0, 0};
};

} // namespace H3D
