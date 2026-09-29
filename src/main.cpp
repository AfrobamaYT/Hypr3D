#include <hyprland/src/plugins/PluginAPI.hpp>
#include <hyprland/src/event/EventBus.hpp>
#include <hyprland/src/render/OpenGL.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/render/pass/PassElement.hpp>
#include <hyprland/src/render/gl/GLFramebuffer.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/desktop/view/Window.hpp>
#include <hyprland/src/managers/SeatManager.hpp>
#include <hyprland/src/managers/eventLoop/EventLoopManager.hpp>
#include <hyprland/src/config/values/ConfigValues.hpp>
#include <hyprland/src/config/ConfigValue.hpp>
#include <hyprland/src/devices/IKeyboard.hpp>
#include <hyprland/src/devices/IPointer.hpp>
#include <hyprland/src/pointer/PointerManager.hpp>
#include <hyprland/src/pointer/PointerController.hpp>
#include <hyprutils/memory/UniquePtr.hpp>

#include <linux/input-event-codes.h>
#include <xkbcommon/xkbcommon.h>

#include "Render/GLScene.hpp"
#include "Input/AimFocus.hpp"
#include "Input/InputController.hpp"
#include "World/World3D.hpp"
#include "HyprlandCompat/WindowsCompat.hpp"
#include "HyprlandCompat/FocusCompat.hpp"
#include "HyprlandCompat/WindowCapture.hpp"
#include "HyprlandCompat/PointerHook.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace H3D {

static HANDLE PHANDLE = nullptr;

static GLScene         g_scene;
static InputController g_input;
static World3D::CWorld g_world;
static AimFocus        g_aim;
static Compat::CWindowCapture g_capture;

static bool g_active = false;

static float g_transition = 0.0f;
static float g_transitionTarget = 0.0f;

static PHLMONITOR g_monitor = nullptr;

static const auto g_inputClockStart = std::chrono::steady_clock::now();
static std::chrono::steady_clock::time_point g_lastTick =
    g_inputClockStart;

static bool g_renderedOnce = false;
static bool g_reportedFramebufferError = false;
static bool g_reportedRenderError = false;
static bool g_reportedPointerHookError = false;

// Layout snapshot taken on entry; empty once the windows are back under the
// layout's control.
static std::vector<Compat::SWindowLayoutSave> g_layoutSaves;
static bool g_ghosted = false;

// What to draw this frame, rebuilt once per frame from the world.
static std::vector<GLScene::WindowRender> g_renderWindows;

// The window that currently owns keyboard focus, as the aim logic last set it.
static std::uintptr_t g_lastFocusId = 0;

// 3D FPS-style mouse gestures. Super is the modifier: LMB moves the aimed
// window in the 3D room, RMB resizes its real Hyprland/Wayland geometry.
enum class EPointerGesture : uint8_t { None, Move3D, ResizeReal };

static bool            g_pointerDown = false;
static EPointerGesture g_pointerGesture = EPointerGesture::None;
static uint32_t        g_pointerButton = 0;
static bool             g_superHeld = false;

static PHLWINDOW        g_clientButtonWindow;
static PHLLS            g_clientButtonLayer;
static Vector2D          g_clientButtonLocal{};
static uint32_t         g_clientButton = 0;
static bool              g_clientButtonDown = false;

struct SResizeGesture {
    bool          active = false;
    std::uintptr_t id = 0;
    PHLWINDOW     window;
    CBox          startBox{};
    Vec3          startCenter{};
    float         startWorldWidth = 0.0f;
    float         startWorldHeight = 0.0f;
    Vec3          planePoint{};
    Vec3          planeNormal{0.0f, 0.0f, 1.0f};
    int           edgeX = 0; // -1 left, +1 right
    int           edgeY = 0; // +1 top, -1 bottom
};

static SResizeGesture g_resize{};

// Guards against a snapshot triggering another copy of the plugin inside
// Hyprland's nested offscreen render.
static bool g_capturing = false;

// Snapshot scheduling: the snapshot pass is the heaviest per-frame cost, so
// windows whose committed buffer and box are unchanged skip it entirely
// (checked inside the capture layer). `force` refreshes regardless -- for the
// aimed window, any gesture target, and the keyboard-focused window (whose
// border styling changes with focus). This is what keeps the room cheap while
// a screen capture (OBS/pipewire) copies every damaged frame.
static uint64_t g_captureFrames = 0;
static std::uintptr_t g_lastAimedId = 0;

// Super+wheel hover zoom: glides the aimed window along the ray from the
// camera through the window toward the target distance (exponential lerp,
// same feel as the drag zoom). Cleared when a drag takes over the window or
// the entity disappears.
static std::uintptr_t s_zoomId     = 0;
static Vec3           s_zoomDir    = {};
static Vec3           s_zoomAnchor = {}; // camera position at the last wheel event
static double         s_zoomCur    = 0.0;
static double         s_zoomTarget = 0.0;

// Windows that appear while the view is open spawn as floating panels of
// this logical size (see ghostWindows), and enter the room this far in front
// of the camera, facing it.
static constexpr float kSpawnWidth    = 960.0f;
static constexpr float kSpawnHeight   = 540.0f;
static constexpr float kSpawnDistance = 10.0f;

// A normal damage cycle stops when nothing else in Hyprland changes. 3D mode is
// itself an animated scene, so keep a small render pump alive while it is open.
// 8 ms targets roughly 120 Hz without making the event loop spin continuously.
static SP<CEventLoopTimer> g_framePump;
static constexpr auto kFramePumpInterval = std::chrono::milliseconds(8);

// Two keyboard modes, toggled with Super + Left Alt:
//   Space  -- WASD/Space/Shift/Ctrl drive the camera;
//   Window -- every key reaches the focused window for typing.
// The mouse behaves identically in both: it always looks around, drags
// windows (Super+LMB), resizes (Super+RMB) and clicks through to clients.
enum class EKeyboardMode : uint8_t { Space, Window };
static EKeyboardMode g_keyboardMode = EKeyboardMode::Space;
static bool          g_altHeld      = false;

// Camera movement keys, set only while the 3D view owns the keyboard.
static bool g_keyFwd = false;
static bool g_keyBack = false;
static bool g_keyLeft = false;
static bool g_keyRight = false;
static bool g_keyUp = false;
static bool g_keyDown = false;
static bool g_keySprint = false;

static bool g_hookInstalled = false;

// Optional panorama background, set via plugin:hypr3d:panorama in the config.
static SP<Config::Values::CStringValue> g_panoramaValue;

static void notify(const std::string& text, const CHyprColor& color);

// Feeds the requested path to the scene every frame. Tilde is expanded, a
// missing file is reported once per path, and the scene itself reloads the
// texture when the file's mtime changes.
static void updatePanorama() {
    if (!g_panoramaValue)
        return;

    std::string path = g_panoramaValue->value();

    if (!path.empty() && path.starts_with('~')) {
        if (const char* HOME = getenv("HOME"))
            path = std::string{HOME} + path.substr(1);
    }

    if (!path.empty()) {
        std::error_code ec;

        if (!std::filesystem::exists(path, ec)) {
            static std::string notified;

            if (notified != path) {
                notified = path;
                notify(
                    "[hypr3d] panorama file not found: " + path,
                    CHyprColor{1.0f, 0.6f, 0.2f, 1.0f}
                );
            }
            return;
        }
    }

    g_scene.setPanoramaPath(path);
}

// --- diagnostics ------------------------------------------------------------
// Written to /tmp/hypr3d-status.txt while the view is live. Both reported
// bugs (mouse not rotating, 2D desktop showing through) admit several
// plausible causes, and guessing between them from the outside cost far more
// time than one dump of the actual numbers.
static uint64_t g_diagMoveEvents = 0;
static uint64_t g_diagSinkCalls   = 0;
static Vector2D g_diagLastPos{};
static double   g_diagLastDx = 0.0;
static double   g_diagLastDy = 0.0;
static Vector2D g_diagPinned{};
static bool     g_diagPinnedValid = false;
static bool     g_warping         = false;
static uint64_t g_diagWarpCalls   = 0;
static Vector2D g_diagWarpAfter{};
static Vector2D g_diagMgrPos{};

// How far (logical px) the cursor may stray from the crosshair before it is
// warped back. Declared with the other diagnostics so the dump and the code
// that uses it cannot drift apart.
static constexpr double kWarpRadiusPx = 8.0;
static float    g_diagAlpha = 0.0f;
static uint64_t g_diagFrames = 0;
static std::chrono::steady_clock::time_point g_diagLastDump{};

static void notify(
    const std::string& text,
    const CHyprColor& color
) {
    if (!PHANDLE)
        return;

    HyprlandAPI::addNotification(
        PHANDLE,
        text,
        color,
        3000
    );
}

static uint32_t inputTimeMs() {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - g_inputClockStart
        ).count();

    return static_cast<uint32_t>(std::max<int64_t>(0, elapsed));
}

static void damageCurrentMonitor() {
    if (!g_monitor)
        return;

    if (!g_pHyprRenderer)
        return;

    g_pHyprRenderer->damageMonitor(
        g_monitor
    );
}

// The monitor the 3D view is built for: whatever has keyboard focus, falling
// back to whatever render.pre last reported.
static PHLMONITOR targetMonitor() {
    if (const auto FOCUSED = Compat::focusedMonitor())
        return FOCUSED;

    return g_monitor;
}

// Where the crosshair sits in logical coordinates: the centre of the monitor
// the view is built for. The pointer is pinned around this point, and the
// picking ray is independent of the cursor and comes from the camera.
static Vector2D crosshairLogical() {
    const auto MON = targetMonitor();

    if (!MON)
        return Vector2D{0.0, 0.0};

    return MON->m_position +
           Vector2D{MON->m_size.x * 0.5, MON->m_size.y * 0.5};
}

static void resetMovementKeys() {
    g_keyFwd = g_keyBack = g_keyLeft = g_keyRight = false;
    g_keyUp = g_keyDown = g_keySprint = false;
    g_superHeld = false;
}

// Clears only the camera keys, keeping the gesture modifiers honest -- used
// when switching into window mode while keys are physically held.
static void resetCameraKeys() {
    g_keyFwd = g_keyBack = g_keyLeft = g_keyRight = false;
    g_keyUp = g_keyDown = g_keySprint = false;
}

// True while the 3D view owns the pointer and the keyboard.
static bool ownsInput() {
    return g_active && g_transition >= 0.9f;
}

static void clearAimFocus() {
    g_aim.reset();
    g_lastFocusId = 0;
    Compat::clearFocus();
    Compat::clearPointerFocus();
}

// --- layout ghosting --------------------------------------------------------
//
// Every window's box is saved first and only then ghosted: ghosting one window
// triggers a relayout of the ones still attached, which would corrupt boxes
// captured afterwards. Ghosting is what stops tiling from managing windows
// while they are living in 3D space.
static void ghostWindows(const PHLMONITOR& mon) {
    if (!mon)
        return;

    const auto INFOS = Compat::enumerateEligibleWindows(mon);

    if (!g_ghosted) {
        // First pass: save EVERY window before ghosting any of them, since
        // ghosting one window triggers a relayout of the ones still attached,
        // which would corrupt boxes captured afterwards.
        g_layoutSaves.clear();
        g_layoutSaves.reserve(INFOS.size());

        for (const auto& info : INFOS) {
            auto SAVE = Compat::saveWindowLayout(info.window);

            if (SAVE.window)
                g_layoutSaves.push_back(std::move(SAVE));
        }

        for (auto& save : g_layoutSaves)
            Compat::applyWindowGhost(save);

        g_ghosted = true;
        return;
    }

    // Steady state: a window that mapped while the view is open must get the
    // same treatment the same frame it becomes eligible, otherwise it stays a
    // live layout target and every later spawn or close re-tiles the space
    // around it. The live weak reference guards against a new window reusing
    // a closed one's address.
    for (const auto& info : INFOS) {
        bool known = false;

        for (const auto& save : g_layoutSaves) {
            if (save.id == info.id && !save.window.expired()) {
                known = true;
                break;
            }
        }

        if (known)
            continue;

        // A window that appears while the view is open becomes a small
        // floating panel instead of a fullscreen tile -- resized BEFORE the
        // layout save, so the box restored on exit is the same 720x480 the
        // user actually worked with.
        if (info.window) {
            const double CX = mon->m_size.x * 0.5 - kSpawnWidth * 0.5;
            const double CY = mon->m_size.y * 0.5 - kSpawnHeight * 0.5;

            Compat::setWindowBox(
                info.window,
                CBox{CX, CY, kSpawnWidth, kSpawnHeight}
            );
        }

        auto SAVE = Compat::saveWindowLayout(info.window);

        if (!SAVE.window)
            continue;

        Compat::applyWindowGhost(SAVE);
        g_layoutSaves.push_back(std::move(SAVE));
    }
}

static void unghostWindows() {
    if (!g_ghosted)
        return;

    // Force the exact saved geometry back, so leaving 3D never disturbs the
    // user's 2D arrangement. The 3D arrangement is a view, not an edit.
    for (auto& save : g_layoutSaves)
        Compat::restoreWindowLayout(save);

    g_layoutSaves.clear();
    g_ghosted = false;
}

// --- continuous 3D frame pump ----------------------------------------------

static void stopFramePump() {
    if (!g_framePump)
        return;

    if (g_pEventLoopManager)
        g_pEventLoopManager->removeTimer(g_framePump);

    g_framePump.reset();
}

static void startFramePump() {
    if (!g_pEventLoopManager)
        return;

    if (!g_framePump) {
        g_framePump = makeShared<CEventLoopTimer>(
            std::nullopt,
            [](SP<CEventLoopTimer> self, void*) {
                if (!g_active) {
                    self->updateTimeout(std::nullopt);
                    return;
                }

                damageCurrentMonitor();
                self->updateTimeout(kFramePumpInterval);
            },
            nullptr
        );

        g_pEventLoopManager->addTimer(g_framePump);
    }

    g_framePump->updateTimeout(kFramePumpInterval);
}

// --- live window capture -----------------------------------------------------

static void refreshCaptures(
    const std::vector<Compat::SWindowInfo>& infos,
    const PHLMONITOR& mon
) {
    if (infos.empty())
        return;

    std::vector<std::uintptr_t> keep;
    keep.reserve(infos.size());

    for (const auto& info : infos)
        keep.push_back(info.id);

    g_capture.retainOnly(keep);

    const auto FOCUSED = Compat::focusedWindow();
    const std::uintptr_t FOCUSED_ID = FOCUSED ? Compat::windowId(FOCUSED) : 0;

    for (const auto& info : infos) {
        // Focus-change feedback (the active/inactive opacity fade and the
        // border color tween) is compositor-side -- no client commit happens,
        // so the buffer-change check would freeze both animations mid-way
        // and make the window snap to its final look. While either runs, the
        // snapshot refreshes at full rate.
        const bool FADING = info.window &&
            (info.window->alpha(Desktop::View::WINDOW_ALPHA_ACTIVE)
                 ->isBeingAnimated() ||
                info.window->m_borderFadeAnimationProgress->isBeingAnimated());

        const bool FORCE =
            FADING ||
            info.id == g_lastAimedId ||
            (g_world.dragActive() && g_world.draggedId() == info.id) ||
            (g_resize.active && g_resize.id == info.id) ||
            info.id == FOCUSED_ID;

        if (info.isLayer)
            g_capture.makeSnapshotLayer(info.layer, mon, FORCE);
        else
            g_capture.makeSnapshot(info.window, mon, FORCE);
    }

    ++g_captureFrames;
}

// Called from render.pre. Hyprland emits render.pre before beginRender() and
// before the compositor's main render pass, so makeSnapshotFB can safely open
// its own offscreen render here. Nested snapshot renders set g_capturing and
// therefore skip all plugin rendering callbacks.
static void serviceCapture() {
    if (!g_active || g_capturing || !g_pHyprRenderer)
        return;

    const auto MON = targetMonitor();
    if (!MON)
        return;

    g_capturing = true;

    ghostWindows(MON);
    refreshCaptures(Compat::enumerateEligibleWindows(MON), MON);

    g_capturing = false;
}

// Handoff the keyboard to whatever the crosshair is on -- and to nothing else.
// Aiming at empty space drops focus entirely rather than leaving the last
// window typed into.
static void updateAimFocus(float dt) {
    if (g_pointerDown || g_clientButtonDown)
        return;

    const auto& cam = g_scene.camera();

    const World3D::SHit HIT =
        g_world.pick(cam.position, cam.centerRay());

    const std::uintptr_t AIMED = HIT.hit ? HIT.id : 0;
    g_lastAimedId = AIMED;

    const std::uintptr_t FOCUS = g_aim.update(AIMED, dt);

    if (FOCUS == g_lastFocusId)
        return;

    const auto WINDOW =
        FOCUS != 0 ? Compat::findWindowById(FOCUS) : nullptr;

    if (WINDOW) {
        Compat::focusWindow(WINDOW);
        g_lastFocusId = FOCUS;
    }
    else {
        Compat::clearFocus();
        g_lastFocusId = 0;
    }
}

static void syncWorld(const PHLMONITOR& mon, float dt) {
    // Deliberately free of side effects on the layout and the renderer. This
    // runs from render.stage, i.e. between the frame's startRenderPass() and
    // endRender(). Ghosting would trigger a relayout while the pass is half
    // built, and makeSnapshotFB opens and closes its OWN nested pass
    // (beginFullFakeRender -> startRenderPass -> endRender), which rebinds the
    // monitor blur FBs; the compositor's outer endRender() then aborted in
    // CMonitor::useFP16(). Capture and ghosting therefore happen outside the
    // frame -- see serviceCapture().
    const auto INFOS = Compat::enumerateEligibleWindows(mon);

    const float MONW = mon->m_size.x;
    const float MONH = mon->m_size.y;

    std::vector<World3D::SEntity> ENTITIES;
    ENTITIES.reserve(INFOS.size());

    for (const auto& info : INFOS) {
        // Not captured yet means not on screen, and something that is not on
        // screen must not be aimable -- otherwise focus could land on an
        // invisible window.
        if (!g_capture.has(info.id))
            continue;

        const auto* SNAPSHOT = g_capture.get(info.id);

        if (!SNAPSHOT)
            continue;

        // Everything below uses the geometry captured WITH the snapshot, not
        // the live box: updateRealResize writes the real window after
        // render.pre, so the live box can be a frame ahead of the captured
        // pixels. Quad, UV subrect and picking all follow the snapshot box,
        // which keeps the drawn content and the crosshair mapping aligned
        // during resizes -- otherwise the edges smear across the frame delta.
        const CBox& BOX = SNAPSHOT->sampledBox;

        World3D::SEntity entity;
        entity.id = info.id;

        // BOX is where the content sits INSIDE the captured texture (it is
        // the real box unless the window was pulled on-screen for the
        // snapshot), so the UV subrect must follow it.
        entity.logicalLeft   = BOX.x;
        entity.logicalTop    = BOX.y;
        entity.logicalWidth  = BOX.w;
        entity.logicalHeight = BOX.h;

        // Where the client surface sits inside the full decorated box. The
        // quad spans content + borders, so surface-local input is the hit
        // position minus this offset -- borders render but never shift input.
        entity.surfaceOffsetX = SNAPSHOT->surfaceOffset.x;
        entity.surfaceOffsetY = SNAPSHOT->surfaceOffset.y;
        entity.surfaceWidth   = SNAPSHOT->surfaceSize.x;
        entity.surfaceHeight  = SNAPSHOT->surfaceSize.y;

        // Seed pose: existing entities own their world position and rotation.
        // NEW windows spawn straight in front of the camera at a fixed read
        // distance, facing it, scaled down to fit within kSpawnFit -- coming
        // in at their full 2D pixel size would fill half the room. The scale
        // is uniform so the captured content never distorts, and it persists
        // for the entity's lifetime (the quad keeps following the box).
        if (const auto* EXISTING = g_world.find(info.id)) {
            entity.center = EXISTING->center;
            entity.yaw = EXISTING->yaw;
            entity.pitch = EXISTING->pitch;
            entity.spawnScale = EXISTING->spawnScale;
        }
        else {
            const auto& CAM = g_scene.camera();
            const Vec3 FWD = CAM.forward();

            entity.center = CAM.position + FWD * kSpawnDistance;

            // Face the camera: with this model's convention (the normal's Y
            // component is -sin(pitch)) the target is the camera's own yaw
            // and pitch.
            entity.yaw   = std::atan2(-FWD.x, -FWD.z);
            entity.pitch = std::asin(std::clamp(FWD.y, -1.0f, 1.0f));

            // The real box is 720x480 (set in ghostWindows before the first
            // snapshot), so the quad follows it at scale 1.
            entity.spawnScale = 1.0f;
        }

        // The quad size ALWAYS follows the snapshot box (times the spawn
        // scale): the UV subrect and the input mapping are box-relative, so a
        // stale world size would squish the content and shrink the input zone
        // on every resize.
        entity.width  = World3D::toWorld(BOX.w) * entity.spawnScale;
        entity.height = World3D::toWorld(BOX.h) * entity.spawnScale;

        if (g_resize.active && g_resize.id == info.id) {
            const float DW = entity.width - g_resize.startWorldWidth;
            const float DH = entity.height - g_resize.startWorldHeight;

            entity.center = g_resize.startCenter;
            entity.center += g_world.rightOf(info.id) *
                (static_cast<float>(g_resize.edgeX) * DW * 0.5f);
            entity.center += g_world.upOf(info.id) *
                (static_cast<float>(g_resize.edgeY) * DH * 0.5f);
        }

        ENTITIES.push_back(entity);
    }

    g_world.setEntities(std::move(ENTITIES), false);

    // --- build the draw list from world + snapshot ---
    g_renderWindows.clear();
    g_renderWindows.reserve(g_world.entities().size());

    for (const auto& entity : g_world.entities()) {
        const auto* SNAPSHOT = g_capture.get(entity.id);

        if (!SNAPSHOT || (SNAPSHOT->texID == 0 && !SNAPSHOT->bigTex))
            continue;

        // The captured texture spans texSpan -- the monitor size, or the
        // enlarged virtual monitor when the window did not fit. Dividing by
        // the live monitor size would rescale an oversized window's content
        // and distort it.
        const float SPANW = SNAPSHOT->texSpan.x;
        const float SPANH = SNAPSHOT->texSpan.y;

        if (SPANW <= 0.0f || SPANH <= 0.0f)
            continue;

        GLScene::WindowRender render;
        render.id      = entity.id;
        render.texture = SNAPSHOT->bigTex ? SNAPSHOT->bigTex : SNAPSHOT->texID;

        render.x = entity.center.x;
        render.y = entity.center.y;
        render.z = entity.center.z;
        render.yaw = entity.yaw;
        render.pitch = entity.pitch;

        render.width  = entity.width;
        render.height = entity.height;

        // The snapshot framebuffer covers the whole monitor, so the window is
        // a subrect of it. Hyprland renders its framebuffers with logical Y
        // (top-left origin, growing down) mapped straight into NDC, so the
        // resulting texture is TOP-DOWN: v = 0 is the monitor's top row and
        // v = logicalY / monH. The quad convention here is v0 at the world
        // bottom of the window, so the content's bottom edge (logicalY =
        // top + height) must feed v0. Flipping with 1 - (...) instead sampled
        // a mirrored band and rendered windows upside down relative to the
        // picked coordinates.
        render.u0 = std::clamp(entity.logicalLeft / SPANW, 0.0f, 1.0f);
        render.u1 = std::clamp(
            (entity.logicalLeft + entity.logicalWidth) / SPANW, 0.0f, 1.0f);
        render.v0 = std::clamp(
            (entity.logicalTop + entity.logicalHeight) / SPANH, 0.0f, 1.0f);
        render.v1 = std::clamp(entity.logicalTop / SPANH, 0.0f, 1.0f);

        g_renderWindows.push_back(render);
    }

    updateAimFocus(dt);
}

// Window-local (top-left origin, logical px) point for a crosshair hit.
static Vector2D localFromHit(const World3D::SHit& hit) {
    const auto* ENTITY = g_world.find(hit.id);
    if (!ENTITY)
        return {};

    // RayHit::u/v span the full decorated box (content + border) with a
    // top-left origin growing downwards, matching Wayland surface-local
    // coordinates. Do not invert them a second time. The client surface sits
    // inside that box at surfaceOffset, so subtracting it converts to
    // surface-local px: content maps 1:1 and the border ring clamps to the
    // surface edge, so drawing borders never shifts input.
    const float X = std::clamp(hit.u, 0.0f, 1.0f) * ENTITY->logicalWidth -
        ENTITY->surfaceOffsetX;
    const float Y = std::clamp(hit.v, 0.0f, 1.0f) * ENTITY->logicalHeight -
        ENTITY->surfaceOffsetY;

    return {
        std::clamp(X, 0.0f, ENTITY->surfaceWidth),
        std::clamp(Y, 0.0f, ENTITY->surfaceHeight),
    };
}

static World3D::SHit aimHit() {
    const auto& cam = g_scene.camera();

    // Nearest hit whose pixels are actually visible: an invisible layer
    // overlay (transparent quickshell PanelWindow) must not swallow the
    // crosshair -- empty pixels fall through to the surface underneath. This
    // is also what keeps aim/focus from flapping between an overlay and the
    // window it covers.
    for (const auto& HIT : g_world.pickAll(cam.position, cam.centerRay())) {
        const auto* SNAPSHOT = g_capture.get(HIT.id);

        if (SNAPSHOT && SNAPSHOT->alphaValid && !SNAPSHOT->alphaMask.empty()) {
            const int MX = std::min(SNAPSHOT->alphaW - 1,
                static_cast<int>(HIT.u * SNAPSHOT->alphaW));
            const int MY = std::min(SNAPSHOT->alphaH - 1,
                static_cast<int>(HIT.v * SNAPSHOT->alphaH));

            const int A = SNAPSHOT->alphaMask[static_cast<size_t>(MY) * SNAPSHOT->alphaW + MX];

            if (A < 8)
                continue; // transparent pixel of an overlay: next surface
        }

        return HIT;
    }

    return {};
}

// Resolves a hit id into either a window or a layer surface.
struct SHitTarget {
    PHLWINDOW window;
    PHLLS     layer;
};

static SHitTarget targetFromHit(std::uintptr_t id) {
    SHitTarget target;

    if (id == 0)
        return target;

    target.layer = Compat::findLayerById(id);

    if (!target.layer)
        target.window = Compat::findWindowById(id);

    return target;
}

static bool rayPlanePoint(
    const Vec3& origin,
    const Vec3& dir,
    const Vec3& planePoint,
    const Vec3& planeNormal,
    Vec3& out
) {
    const float denom = dot(dir, planeNormal);
    if (!std::isfinite(denom) || std::fabs(denom) < 0.000001f)
        return false;

    const float t = dot(planePoint - origin, planeNormal) / denom;
    if (!std::isfinite(t) || t <= 0.0f)
        return false;

    out = origin + dir * t;
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z);
}

static Vector2D worldPointToGlobalPx(
    const PHLMONITOR& mon,
    const Vec3& point
) {
    (void)mon;

    const Vec3 LOCAL = g_world.localPoint(g_resize.id, point);
    const CBox CURRENT = Compat::currentWindowBox(g_resize.window);

    const double currentCenterX = CURRENT.x + CURRENT.w * 0.5;
    const double currentCenterY = CURRENT.y + CURRENT.h * 0.5;

    return {
        currentCenterX + static_cast<double>(LOCAL.x) *
            World3D::LOGICAL_PX_PER_UNIT,
        currentCenterY - static_cast<double>(LOCAL.y) *
            World3D::LOGICAL_PX_PER_UNIT,
    };
}

static CBox resizeBoxFromAim(const Vec3& point, const PHLMONITOR& mon) {
    const double X = worldPointToGlobalPx(mon, point).x;
    const double Y = worldPointToGlobalPx(mon, point).y;

    const double RIGHT = g_resize.startBox.x + g_resize.startBox.w;
    const double BOTTOM = g_resize.startBox.y + g_resize.startBox.h;

    CBox out = g_resize.startBox;

    if (g_resize.edgeX > 0) {
        out.x = g_resize.startBox.x;
        out.w = X - out.x;
    } else {
        out.x = X;
        out.w = RIGHT - out.x;
    }

    if (g_resize.edgeY > 0) {
        out.y = Y;
        out.h = BOTTOM - out.y;
    } else {
        out.y = g_resize.startBox.y;
        out.h = Y - out.y;
    }

    const auto MIN = g_resize.window->minSize().value_or(Vector2D{1.0, 1.0});
    const auto MAX = g_resize.window->maxSize().value_or(Vector2D{INFINITY, INFINITY});

    out.w = std::clamp(out.w, MIN.x, MAX.x);
    out.h = std::clamp(out.h, MIN.y, MAX.y);

    if (g_resize.edgeX > 0)
        out.x = g_resize.startBox.x;
    else
        out.x = RIGHT - out.w;

    if (g_resize.edgeY > 0)
        out.y = BOTTOM - out.h;
    else
        out.y = g_resize.startBox.y;

    return out;
}

static void updateRealResize() {
    if (!g_resize.active || !g_resize.window)
        return;

    const auto MON = targetMonitor();
    if (!MON)
        return;

    const auto* ENTITY = g_world.find(g_resize.id);
    if (!ENTITY)
        return;

    // The edge being dragged is part of the real, resized window, so the
    // interaction plane follows the window's current centre every frame.
    g_resize.planePoint = ENTITY->center;
    g_resize.planeNormal = g_world.normalOf(g_resize.id);

    Vec3 point;
    if (!rayPlanePoint(
            g_scene.camera().position,
            g_scene.camera().centerRay(),
            g_resize.planePoint,
            g_resize.planeNormal,
            point))
        return;

    const CBox BOX = resizeBoxFromAim(point, MON);
    Compat::setWindowBox(g_resize.window, BOX);
}

static void resetPointerGesture() {
    g_pointerDown = false;
    g_pointerGesture = EPointerGesture::None;
    g_pointerButton = 0;
    g_resize = {};
}

static void finishClientButton(uint32_t timeMs) {
    if (!g_clientButtonDown)
        return;

    if (g_clientButtonLayer) {
        Compat::deliverClick(
            g_clientButtonLayer,
            g_clientButtonLocal,
            g_clientButton,
            false,
            timeMs
        );
    } else if (g_clientButtonWindow) {
        Compat::deliverClick(
            g_clientButtonWindow,
            g_clientButtonLocal,
            g_clientButton,
            false,
            timeMs
        );
    }

    g_clientButtonWindow = nullptr;
    g_clientButtonLayer  = nullptr;
    g_clientButton = 0;
    g_clientButtonDown = false;
}

static void forwardPointerToAim(uint32_t timeMs) {
    if (!ownsInput() || g_pointerDown)
        return;

    const auto HIT = aimHit();
    const auto TARGET = HIT.hit ? targetFromHit(HIT.id) : SHitTarget{};

    if (g_clientButtonDown && (g_clientButtonWindow || g_clientButtonLayer)) {
        // A pressed client keeps pointer ownership while the camera turns. If
        // the crosshair leaves that surface, hold the last local point instead
        // of teleporting pointer focus into a different client mid-drag.
        const bool SAME =
            (g_clientButtonWindow && TARGET.window == g_clientButtonWindow) ||
            (g_clientButtonLayer && TARGET.layer == g_clientButtonLayer);

        if (SAME && HIT.hit)
            g_clientButtonLocal = localFromHit(HIT);

        if (g_clientButtonLayer)
            Compat::deliverMotion(g_clientButtonLayer, g_clientButtonLocal, timeMs);
        else if (g_clientButtonWindow)
            Compat::deliverMotion(g_clientButtonWindow, g_clientButtonLocal, timeMs);
        return;
    }

    if (TARGET.layer) {
        Compat::deliverMotion(TARGET.layer, localFromHit(HIT), timeMs);
        return;
    }

    if (!TARGET.window) {
        Compat::clearPointerFocus();
        return;
    }

    Compat::deliverMotion(TARGET.window, localFromHit(HIT), timeMs);
}

// Absolute-position fallback for installations where the PointerManager hook
// is unavailable. When the hook works, it feeds onPointerMotion directly and
// this event only keeps the baseline warm.
static bool onPointerMotion(double dx, double dy) {
    if (!ownsInput())
        return false;

    g_input.addMotion(dx, dy);
    damageCurrentMonitor();
    return true;
}

static void onRenderPre(PHLMONITOR mon) {
    if (g_capturing)
        return;

    if (!g_active) {
        g_monitor = mon;
        return;
    }

    const auto FOCUSED = Compat::focusedMonitor();
    if (FOCUSED && FOCUSED != mon)
        return;

    g_monitor = mon;
    serviceCapture();
}


// --- lifecycle --------------------------------------------------------------

static void deactivate3D() {
    finishClientButton(0);
    resetPointerGesture();

    g_active = false;
    stopFramePump();

    // Restore the host cursor before anything else touches focus: removing
    // pointer focus makes the client re-apply its own cursor image on the
    // next pointer enter, and the default shape shows immediately.
    if (Compat::setCursorHidden(false) && g_pHyprRenderer)
        g_pHyprRenderer->setCursorFromName("default", true);

    clearAimFocus();
    g_world.clear();
    g_renderWindows.clear();
    g_input.reset();
    resetMovementKeys();

    g_capture.releaseAll();
    Compat::setPointerCapture(false);

    unghostWindows();
    g_renderedOnce = false;
}

static void enter3D() {
    // Without this the render stage bails out immediately and the toggle does
    // nothing at all.
    g_active = true;
    startFramePump();

    // Hide the host cursor: the 3D view aims with its own crosshair. Client
    // cursor updates are gated by the hooks; the client re-applies its own
    // image automatically when pointer focus re-enters on exit.
    if (!Compat::setCursorHidden(true))
        notify(
            "[hypr3d] cursor hooks unavailable: the frozen cursor will stay visible",
            CHyprColor{1.0f, 0.6f, 0.2f, 1.0f}
        );
    else
        Pointer::mgr()->resetCursorImage();

    // Focus is absent by default: it only ever appears when the crosshair is
    // on a window.
    clearAimFocus();

    finishClientButton(0);
    resetPointerGesture();

    g_scene.reset();
    g_input.reset();
    g_world.clear();
    g_renderWindows.clear();

    g_capture.releaseAll();

    resetMovementKeys();

    g_keyboardMode = EKeyboardMode::Space;
    g_altHeld      = false;
    s_zoomId       = 0;

    g_renderedOnce = false;
    g_captureFrames = 0;
    g_reportedFramebufferError = false;
    g_reportedRenderError = false;

    // The first full live capture is performed from render.pre, where
    // Hyprland has not entered its main render pass yet.
}

static void toggle3D() {
    g_transitionTarget =
        g_transitionTarget > 0.5f ? 0.0f : 1.0f;

    if (g_transitionTarget > 0.5f)
        enter3D();

    damageCurrentMonitor();

    notify(
        "[hypr3d] toggle",
        CHyprColor{0.2f, 0.8f, 0.4f, 1.0f}
    );
}

static void open3D() {
    g_transitionTarget = 1.0f;
    enter3D();
    damageCurrentMonitor();

    notify(
        "[hypr3d] open",
        CHyprColor{0.2f, 0.8f, 0.4f, 1.0f}
    );
}

static void close3D() {
    g_transitionTarget = 0.0f;
    damageCurrentMonitor();

    notify(
        "[hypr3d] close",
        CHyprColor{0.2f, 0.8f, 0.4f, 1.0f}
    );
}

static float updateTransition() {
    const auto now = std::chrono::steady_clock::now();

    const float dt =
        std::chrono::duration<float>(now - g_lastTick).count();

    g_lastTick = now;

    constexpr float duration = 0.55f;
    constexpr float speed = 1.0f / duration;

    if (g_transition < g_transitionTarget)
        g_transition = std::min(g_transition + dt * speed, 1.0f);
    else if (g_transition > g_transitionTarget)
        g_transition = std::max(g_transition - dt * speed, 0.0f);

    return std::clamp(dt, 0.0f, 0.1f);
}

// --- frame ------------------------------------------------------------------

static void applyCameraMovement(float dt) {
    const float FORWARD  = (g_keyFwd ? 1.f : 0.f) - (g_keyBack ? 1.f : 0.f);
    const float STRAFE   = (g_keyRight ? 1.f : 0.f) - (g_keyLeft ? 1.f : 0.f);
    const float VERTICAL = (g_keyUp ? 1.f : 0.f) - (g_keyDown ? 1.f : 0.f);

    auto& CAM = g_scene.camera();

    if (FORWARD == 0.f && STRAFE == 0.f && VERTICAL == 0.f)
        return;

    const float BASE = CAM.moveSpeed;

    CAM.moveSpeed = BASE * (g_keySprint ? 2.5f : 1.0f);
    CAM.move(FORWARD, STRAFE, VERTICAL, dt);
    CAM.moveSpeed = BASE;

    damageCurrentMonitor();
}

static void update3D(float dt) {
    // Release capture unconditionally when the view no longer owns input, so
    // a disappearing monitor or a closing transition can never strand the
    // pointer in captured mode.
    Compat::setPointerCapture(g_hookInstalled && ownsInput());

    float yawDelta = 0.0f;
    float pitchDelta = 0.0f;

    if (g_input.consumeLook(yawDelta, pitchDelta)) {
        g_scene.rotateView(yawDelta, pitchDelta);
        damageCurrentMonitor();
    }

    applyCameraMovement(dt);

    // decoration:blur feeds the frost overlay on transparent windows
    // (CConfigValue binds by name and works with both hyprland.conf and
    // hyprland.lua -- the same accessor Hyprland's own renderer uses).
    {
        static const CConfigValue<Config::INTEGER> PBLURENABLED("decoration:blur:enabled");
        static const CConfigValue<Config::INTEGER> PBLURSIZE("decoration:blur:size");
        static const CConfigValue<Config::INTEGER> PBLURPASSES("decoration:blur:passes");
        static const CConfigValue<Config::FLOAT>   PBLURVIBRANCY("decoration:blur:vibrancy");

        g_scene.setBlurConfig(
            *PBLURENABLED != 0,
            static_cast<int>(std::clamp(*PBLURSIZE, int64_t{1}, int64_t{32})),
            static_cast<int>(std::clamp(*PBLURPASSES, int64_t{1}, int64_t{8})),
            *PBLURVIBRANCY);
    }

    // Super+wheel hover zoom: glide the window along its ray toward the
    // target distance.
    if (s_zoomId != 0) {
        if (auto* ZOOMED = g_world.find(s_zoomId)) {
            s_zoomCur += (s_zoomTarget - s_zoomCur) *
                (1.0 - std::exp(-8.0 * dt));

            // Glide along the ray ANCHORED at the last wheel event: the
            // window stays world-fixed instead of riding the camera.
            ZOOMED->center = s_zoomAnchor +
                s_zoomDir * static_cast<float>(s_zoomCur);

            // One-shot glide: once the target is reached the state is
            // dropped and the window is world-fixed again -- a live zoom
            // state would keep the window glued to a moving camera.
            if (std::fabs(s_zoomTarget - s_zoomCur) < 0.02)
                s_zoomId = 0;
        } else {
            s_zoomId = 0;
        }
    }

    if (g_transition <= 0.0f)
        return;

    const auto MON = targetMonitor();

    if (!MON)
        return;

    updatePanorama();

    // The camera pose has changed for this frame. Use that same new centre ray
    // for the active gesture, so movement and resize track exactly what the
    // user is looking at rather than the previous frame's ray.
    if (g_pointerGesture == EPointerGesture::Move3D && g_pointerDown) {
        g_world.updateDrag(
            g_scene.camera().position,
            g_scene.camera().centerRay(),
            dt
        );
    } else if (g_pointerGesture == EPointerGesture::ResizeReal && g_pointerDown) {
        updateRealResize();
    }

    syncWorld(MON, dt);

    // Normal client interaction is a virtual pointer located exactly at the
    // crosshair. It is updated every frame after camera motion, so buttons,
    // text fields, scrollbars, etc. receive ordinary Wayland pointer motion.
    if (!g_pointerDown)
        forwardPointerToAim(inputTimeMs());
}

class CHypr3DPassElement final : public IPassElement {
  public:
    CHypr3DPassElement(float alpha, float dt) :
        m_alpha(alpha), m_dt(dt) {}

    std::vector<UP<IPassElement>> draw() override {
        if (!g_pHyprRenderer)
            return {};

        if (g_pHyprRenderer->type() != Render::IHyprRenderer::RT_GL) {
            reportFramebufferErrorOnce("[hypr3d] renderer is not OpenGL");
            return {};
        }

        auto& renderData = g_pHyprRenderer->m_renderData;

        if (!renderData.currentFB) {
            reportFramebufferErrorOnce("[hypr3d] current framebuffer is null");
            return {};
        }

        auto* framebuffer =
            dynamic_cast<Render::GL::CGLFramebuffer*>(
                renderData.currentFB.get()
            );

        if (!framebuffer) {
            reportFramebufferErrorOnce(
                "[hypr3d] current framebuffer is not CGLFramebuffer"
            );
            return {};
        }

        const int width = std::max(
            1,
            static_cast<int>(std::round(renderData.currentFB->m_size.x))
        );

        const int height = std::max(
            1,
            static_cast<int>(std::round(renderData.currentFB->m_size.y))
        );

        const GLuint framebufferID = framebuffer->getFBID();

        if (framebufferID == 0) {
            reportFramebufferErrorOnce(
                "[hypr3d] current framebuffer has ID 0"
            );
            return {};
        }

        if (!Render::GL::g_pHyprOpenGL) {
            reportFramebufferErrorOnce("[hypr3d] g_pHyprOpenGL is null");
            return {};
        }

        Render::GL::g_pHyprOpenGL->makeEGLCurrent();

        const bool result = g_scene.render(
            framebufferID,
            width,
            height,
            m_alpha,
            m_dt,
            g_renderWindows
        );

        if (!result) {
            if (!g_reportedRenderError) {
                g_reportedRenderError = true;
                notify(
                    "[hypr3d] GLScene::render failed",
                    CHyprColor{1.0f, 0.2f, 0.2f, 1.0f}
                );
            }

            return {};
        }

        if (!g_renderedOnce) {
            g_renderedOnce = true;

            notify(
                "[hypr3d] 3D pass executed " +
                    std::to_string(width) + "x" + std::to_string(height),
                CHyprColor{0.2f, 1.0f, 0.5f, 1.0f}
            );
        }

        return {};
    }

    bool needsLiveBlur() override {
        return false;
    }

    bool needsPrecomputeBlur() override {
        return false;
    }

    const char* passName() override {
        return "Hypr3D";
    }

    ePassElementType type() override {
        return EK_CUSTOM;
    }

    bool undiscardable() override {
        return true;
    }

    bool disableSimplification() override {
        return true;
    }

    std::optional<CBox> boundingBox() override {
        return std::nullopt;
    }

    CRegion opaqueRegion() override {
        return {};
    }

  private:
    static void reportFramebufferErrorOnce(const std::string& text) {
        if (g_reportedFramebufferError)
            return;

        g_reportedFramebufferError = true;

        notify(text, CHyprColor{1.0f, 0.2f, 0.2f, 1.0f});
    }

    float m_alpha;
    float m_dt;
};

// Throttled snapshot of the live state, written where I can read it directly
// instead of asking the user to describe what they see.
static void dumpStatus() {
    const auto NOW = std::chrono::steady_clock::now();

    if (std::chrono::duration<float>(NOW - g_diagLastDump).count() < 0.25f)
        return;

    g_diagLastDump = NOW;
    ++g_diagFrames;

    // Requested here, filled in later this frame by the pass element, read by
    // the next dump -- one frame of lag on a value that is not changing.
    g_scene.requestProbe();

    std::ofstream out("/tmp/hypr3d-status.txt", std::ios::trunc);

    if (!out)
        return;

    const auto MON = targetMonitor();

    out << "active=" << (g_active ? 1 : 0) << " frames=" << g_diagFrames
        << " transition=" << g_transition
        << " target=" << g_transitionTarget
        << " alphaSent=" << g_diagAlpha
        << " renderedOnce=" << (g_renderedOnce ? 1 : 0) << "\n";

    out << "hookInstalled=" << (g_hookInstalled ? 1 : 0)
        << " hookActive=" << (Compat::pointerHookActive() ? 1 : 0)
        << " sinkCalls=" << g_diagSinkCalls
        << " moveEvents=" << g_diagMoveEvents << "\n";

    out << "lastPos=" << g_diagLastPos.x << "," << g_diagLastPos.y
        << " pinned=" << g_diagPinned.x << "," << g_diagPinned.y
        << " lastDelta=" << g_diagLastDx << "," << g_diagLastDy << "\n";

    // warpAfter is what the compositor reported immediately after the last
    // warp: equal to `center` means the pin sticks, equal to the pre-warp
    // position means warpTo is a no-op here.
    out << "warpCalls=" << g_diagWarpCalls
        << " warpAfter=" << g_diagWarpAfter.x << "," << g_diagWarpAfter.y
        << " warpRadius=" << kWarpRadiusPx << "\n";

    // mgrPos vs lastPos: if these differ persistently, the event's position
    // and Pointer::mgr()'s position are different quantities and only one of
    // them can be used to derive a delta.
    out << "mgrPos=" << g_diagMgrPos.x << "," << g_diagMgrPos.y << "\n";

    const Vector2D CENTER = crosshairLogical();

    if (MON)
        out << "monitor=" << MON->m_name
            << " pos=" << MON->m_position.x << "," << MON->m_position.y
            << " size=" << MON->m_size.x << "x" << MON->m_size.y
            << " pixel=" << MON->m_pixelSize.x << "x" << MON->m_pixelSize.y
            << " center=" << CENTER.x << "," << CENTER.y << "\n";
    else
        out << "monitor=none\n";

    // The decisive number for the "2D desktop showing through" report: the
    // alpha actually sitting in the offscreen scene buffer at screen centre.
    if (g_scene.probeValid()) {
        const unsigned char* P = g_scene.probeRGBA();
        out << "scenePixel=" << int(P[0]) << "," << int(P[1]) << ","
            << int(P[2]) << "," << int(P[3]) << "\n";
    }
    else
        out << "scenePixel=unavailable\n";

    out << "renderWindows=" << g_renderWindows.size() << "\n";
    out << "captureFrames=" << g_captureFrames << "\n";
}

static void onRenderStage(eRenderStage stage) {
    if (g_capturing)
        return;

    if (!g_active)
        return;

    if (!g_monitor)
        return;

    if (stage != RENDER_LAST_MOMENT)
        return;

    dumpStatus();

    const float dt = updateTransition();

    if (g_transition <= 0.0f && g_transitionTarget <= 0.0f) {
        deactivate3D();
        return;
    }

    if (!g_pHyprRenderer)
        return;

    if (g_pHyprRenderer->type() != Render::IHyprRenderer::RT_GL)
        return;

    update3D(dt);

    g_diagAlpha = std::clamp(g_transition, 0.0f, 1.0f);

    g_pHyprRenderer->addPassElement(
        makeUnique<CHypr3DPassElement>(
            g_diagAlpha,
            dt
        )
    );

    damageCurrentMonitor();
}

// --- event handlers ---------------------------------------------------------

// The pointer hook gives us true relative motion and consumes it, so the OS
// cursor remains parked and cannot hit a screen edge. If the hook is missing,
// the absolute event path derives a delta and keeps the camera usable.
static bool hookSink(double dx, double dy) {
    ++g_diagSinkCalls;
    return onPointerMotion(dx, dy);
}

static void onMouseMove(Vector2D pos, Event::SCallbackInfo& info) {
    ++g_diagMoveEvents;
    g_diagLastPos = pos;

    if (!ownsInput())
        return;

    if (g_hookInstalled && g_diagSinkCalls > 0) {
        // The relative hook already owns the real delta. The absolute event is
        // intentionally ignored for rotation so it cannot double-count it.
        info.cancelled = true;
        return;
    }

    if (!g_diagPinnedValid) {
        g_diagPinned = pos;
        g_diagPinnedValid = true;
        info.cancelled = true;
        return;
    }

    const double dx = pos.x - g_diagPinned.x;
    const double dy = pos.y - g_diagPinned.y;
    g_diagPinned = pos;

    onPointerMotion(dx, dy);
    info.cancelled = true;
}

static void onMouseAxis(
    IPointer::SAxisEvent event,
    Event::SCallbackInfo& info
) {
    if (!ownsInput())
        return;

    if (event.axis != WL_POINTER_AXIS_VERTICAL_SCROLL)
        return;

    // Physical wheels report whole detents in deltaDiscrete; touchpads send
    // a smooth delta instead. Hi-res wheels report many detents per physical
    // click -- clamped, or a light scroll flings the window across the room.
    const double STEPS = std::clamp(
        event.deltaDiscrete != 0 ?
            static_cast<double>(event.deltaDiscrete) :
            event.delta * 0.05,
        -2.0, 2.0);

    if (STEPS == 0.0)
        return;

    // During an LMB drag the wheel zooms the dragged window.
    if (g_pointerGesture == EPointerGesture::Move3D && g_pointerDown) {
        g_world.dragZoom(STEPS);
        info.cancelled = true;
        damageCurrentMonitor();
        return;
    }

    // Any other time: Super + wheel over a window zooms it along the ray
    // from the camera through the window. Without Super the wheel reaches
    // the focused client unchanged (scroll).
    if (!g_superHeld)
        return;

    const auto HIT = aimHit();

    if (!HIT.hit)
        return;

    const auto* ENTITY = g_world.find(HIT.id);

    if (!ENTITY)
        return;

    const auto& CAM = g_scene.camera();
    const Vec3 OFFSET = ENTITY->center - CAM.position;
    const float DIST = std::sqrt(
        OFFSET.x * OFFSET.x + OFFSET.y * OFFSET.y + OFFSET.z * OFFSET.z);

    if (!std::isfinite(DIST) || DIST < 0.05f)
        return;

    const bool NEW = s_zoomId != HIT.id;
    s_zoomId     = HIT.id;
    s_zoomDir    = OFFSET * (1.0f / DIST);
    s_zoomAnchor = CAM.position;

    if (NEW) {
        s_zoomCur    = DIST;
        s_zoomTarget = DIST;
    }

    s_zoomTarget = std::clamp(
        s_zoomTarget * std::pow(1.06, STEPS), 2.0, 40.0);

    info.cancelled = true;
    damageCurrentMonitor();
}

static void onMouseButton(
    IPointer::SButtonEvent event,
    Event::SCallbackInfo& info
) {
    if (!ownsInput())
        return;

    const bool PRESSED =
        event.state == WL_POINTER_BUTTON_STATE_PRESSED;

    // Release the plugin gesture that owns this physical button.
    if (!PRESSED && g_pointerDown && event.button == g_pointerButton) {
        resetPointerGesture();
        info.cancelled = true;
        damageCurrentMonitor();
        return;
    }

    // Super+LMB / Super+RMB are exclusively plugin gestures. Nothing is sent
    // to the client for these physical button events. Layer surfaces can be
    // dragged like windows, but not resized -- their real geometry is owned
    // by the shell that anchored them.
    if (PRESSED && g_superHeld &&
        (event.button == BTN_LEFT || event.button == BTN_RIGHT)) {
        const World3D::SHit HIT = aimHit();
        const auto TARGET = HIT.hit ? targetFromHit(HIT.id) : SHitTarget{};

        if (!TARGET.window && !TARGET.layer) {
            info.cancelled = true;
            return;
        }

        const auto& CAM = g_scene.camera();

        if (TARGET.window)
            Compat::focusWindow(TARGET.window);

        if (event.button == BTN_RIGHT && !TARGET.window) {
            info.cancelled = true;
            return;
        }

        if (event.button == BTN_LEFT) {
            if (!g_world.startDrag(
                    HIT.id, HIT, CAM.position, CAM.forward())) {
                info.cancelled = true;
                return;
            }
            s_zoomId = 0; // the drag owns this window's distance now
            g_pointerGesture = EPointerGesture::Move3D;
            g_pointerButton = BTN_LEFT;
            g_pointerDown = true;
            g_resize = {};
        }
        else {
            const auto ENTITY = g_world.find(HIT.id);
            if (!ENTITY) {
                info.cancelled = true;
                return;
            }

            g_resize = {};
            g_resize.active = true;
            g_resize.id = HIT.id;
            g_resize.window = TARGET.window;
            g_resize.startBox = Compat::currentWindowBox(TARGET.window);
            g_resize.startCenter = ENTITY->center;
            g_resize.startWorldWidth = ENTITY->width;
            g_resize.startWorldHeight = ENTITY->height;
            g_resize.planePoint = ENTITY->center;
            g_resize.planeNormal = g_world.normalOf(HIT.id);

            // The point currently under the crosshair chooses the corner that
            // will follow it. The opposite corner is fixed. As the camera
            // turns, the current centre ray is intersected with this same
            // window plane, so the real window stretches exactly toward the
            // point being aimed at.
            // RayHit::v is already top-to-bottom. Top half follows +1,
            // bottom half follows -1 in the CBox edge convention below.
            g_resize.edgeX = HIT.u < 0.5f ? -1 : 1;
            g_resize.edgeY = HIT.v < 0.5f ? 1 : -1;

            g_pointerGesture = EPointerGesture::ResizeReal;
            g_pointerButton = BTN_RIGHT;
            g_pointerDown = true;
        }

        info.cancelled = true;
        damageCurrentMonitor();
        return;
    }

    // Regular buttons are virtual-pointer buttons at the crosshair. Track the
    // pressed surface so release goes back to the same client even if the
    // camera turns away before release.
    const World3D::SHit HIT = aimHit();
    const auto TARGET = HIT.hit ? targetFromHit(HIT.id) : SHitTarget{};

    if (!TARGET.window && !TARGET.layer) {
        info.cancelled = true;
        return;
    }

    const Vector2D LOCAL = localFromHit(HIT);

    if (PRESSED) {
        if (TARGET.window)
            Compat::focusWindow(TARGET.window);

        if (TARGET.layer)
            Compat::deliverClick(TARGET.layer, LOCAL, event.button, true, event.timeMs);
        else
            Compat::deliverClick(TARGET.window, LOCAL, event.button, true, event.timeMs);

        g_clientButtonWindow = TARGET.window;
        g_clientButtonLayer  = TARGET.layer;
        g_clientButton = event.button;
        g_clientButtonLocal = LOCAL;
        g_clientButtonDown = true;
    }
    else {
        if (g_clientButtonDown && (g_clientButtonWindow || g_clientButtonLayer)) {
            if (g_clientButtonLayer)
                Compat::deliverClick(
                    g_clientButtonLayer,
                    g_clientButtonLocal,
                    g_clientButton,
                    false,
                    event.timeMs
                );
            else
                Compat::deliverClick(
                    g_clientButtonWindow,
                    g_clientButtonLocal,
                    g_clientButton,
                    false,
                    event.timeMs
                );
            g_clientButtonWindow = nullptr;
            g_clientButtonLayer  = nullptr;
            g_clientButton = 0;
            g_clientButtonDown = false;
        }
    }

    info.cancelled = true;
    damageCurrentMonitor();
}

// Space mode: only the camera movement keys are swallowed -- everything else
// still reaches the window the crosshair is aiming at. Minecraft creative-
// flight scheme: Space ascends, Shift descends, Ctrl sprints.
static bool isMovementSym(xkb_keysym_t sym) {
    switch (sym) {
        case XKB_KEY_w:
        case XKB_KEY_a:
        case XKB_KEY_s:
        case XKB_KEY_d:
        case XKB_KEY_space:
        case XKB_KEY_Shift_L:
        case XKB_KEY_Shift_R:
        case XKB_KEY_Control_L:
        case XKB_KEY_Control_R:
        case XKB_KEY_Super_L:
        case XKB_KEY_Super_R: return true;
        default: return false;
    }
}

static void setMovementSym(xkb_keysym_t sym, bool down) {
    switch (sym) {
        case XKB_KEY_w: g_keyFwd = down; break;
        case XKB_KEY_s: g_keyBack = down; break;
        case XKB_KEY_a: g_keyLeft = down; break;
        case XKB_KEY_d: g_keyRight = down; break;
        case XKB_KEY_space: g_keyUp = down; break;
        case XKB_KEY_Shift_L:
        case XKB_KEY_Shift_R: g_keyDown = down; break;
        case XKB_KEY_Control_L:
        case XKB_KEY_Control_R: g_keySprint = down; break;
        default: break;
    }
}

static void onKeyboardKey(
    IKeyboard::SKeyEvent event,
    Event::SCallbackInfo& info
) {
    if (!ownsInput())
        return;

    if (!g_pSeatManager || g_pSeatManager->m_keyboard.expired())
        return;

    const auto KEYBOARD = g_pSeatManager->m_keyboard.lock();

    if (!KEYBOARD || !KEYBOARD->m_xkbSymState)
        return;

    // libinput reports evdev codes; xkb wants them offset by 8.
    const auto SYM = xkb_state_key_get_one_sym(
        KEYBOARD->m_xkbSymState,
        event.keycode + 8
    );

    const bool PRESSED = event.state == WL_KEYBOARD_KEY_STATE_PRESSED;

    // Modifier state is tracked in both modes: Super drives the mouse
    // gestures, and Super + Left Alt toggles the keyboard mode.
    if (SYM == XKB_KEY_Super_L || SYM == XKB_KEY_Super_R)
        g_superHeld = PRESSED;
    else if (SYM == XKB_KEY_Alt_L)
        g_altHeld = PRESSED;

    if (PRESSED && g_superHeld && g_altHeld &&
        (SYM == XKB_KEY_Alt_L || SYM == XKB_KEY_Super_L)) {
        g_keyboardMode =
            g_keyboardMode == EKeyboardMode::Space ?
                EKeyboardMode::Window :
                EKeyboardMode::Space;

        if (g_keyboardMode == EKeyboardMode::Window)
            resetCameraKeys(); // held camera keys must not keep walking

        notify(
            g_keyboardMode == EKeyboardMode::Space ?
                "[hypr3d] keyboard: space (wasd / space / shift / ctrl)" :
                "[hypr3d] keyboard: window (typing reaches the focused window)",
            CHyprColor{0.2f, 0.8f, 0.4f, 1.0f}
        );

        info.cancelled = true;
        return;
    }

    // Window mode: every key reaches the focused window untouched.
    if (g_keyboardMode == EKeyboardMode::Window)
        return;

    if (!isMovementSym(SYM))
        return;

    setMovementSym(SYM, PRESSED);

    info.cancelled = true;
}

// --- plugin entry -----------------------------------------------------------

static int luaToggle(lua_State*) {
    toggle3D();
    return 0;
}

static int luaOpen(lua_State*) {
    open3D();
    return 0;
}

static int luaClose(lua_State*) {
    close3D();
    return 0;
}

static SDispatchResult dispatchToggle(std::string) {
    toggle3D();
    return {};
}

static SDispatchResult dispatchOpen(std::string) {
    open3D();
    return {};
}

static SDispatchResult dispatchClose(std::string) {
    close3D();
    return {};
}

APICALL EXPORT std::string PLUGIN_API_VERSION() {
    return HYPRLAND_API_VERSION;
}

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE handle) {
    PHANDLE = handle;

    const std::string serverHash = __hyprland_api_get_hash();
    const std::string clientHash = __hyprland_api_get_client_hash();

    if (serverHash != clientHash)
        throw std::runtime_error("[hypr3d] Hyprland API hash mismatch");

    // Relative pointer capture is what lets the mouse look around freely
    // instead of hitting the edge of whatever screen it started on.
    g_hookInstalled =
        Compat::installPointerHook(PHANDLE, &hookSink);

    if (!g_hookInstalled) {
        g_reportedPointerHookError = true;

        notify(
            "[hypr3d] pointer hook unavailable: mouse-look falls back to "
            "cursor-relative motion and will stop at screen edges",
            CHyprColor{1.0f, 0.6f, 0.2f, 1.0f}
        );
    }

    HyprlandAPI::addNotification(
        PHANDLE,
        "[hypr3d] renderer loaded",
        CHyprColor{0.2f, 0.8f, 0.4f, 1.0f},
        3000
    );

    if (!HyprlandAPI::addDispatcherV2(PHANDLE, "hypr3d:toggle", dispatchToggle))
        throw std::runtime_error("[hypr3d] failed to register toggle dispatcher");

    if (!HyprlandAPI::addDispatcherV2(PHANDLE, "hypr3d:open", dispatchOpen))
        throw std::runtime_error("[hypr3d] failed to register open dispatcher");

    if (!HyprlandAPI::addDispatcherV2(PHANDLE, "hypr3d:close", dispatchClose))
        throw std::runtime_error("[hypr3d] failed to register close dispatcher");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "toggle", luaToggle))
        throw std::runtime_error("[hypr3d] failed to register Lua toggle");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "open", luaOpen))
        throw std::runtime_error("[hypr3d] failed to register Lua open");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "close", luaClose))
        throw std::runtime_error("[hypr3d] failed to register Lua close");

    g_panoramaValue = makeShared<Config::Values::CStringValue>(
        "plugin:hypr3d:panorama",
        "3D room panorama background: equirectangular image path",
        Config::STRING{},
        Config::Values::SStringValueOptions{}
    );

    if (!HyprlandAPI::addConfigValueV2(PHANDLE, g_panoramaValue))
        notify(
            "[hypr3d] failed to register plugin:hypr3d:panorama",
            CHyprColor{1.0f, 0.6f, 0.2f, 1.0f}
        );

    static auto renderPre =
        Event::bus()->m_events.render.pre.listen(
            [](PHLMONITOR mon) { onRenderPre(mon); }
        );

    static auto renderStage =
        Event::bus()->m_events.render.stage.listen(
            [](eRenderStage stage) { onRenderStage(stage); }
        );

    static auto mouseMove =
        Event::bus()->m_events.input.mouse.move.listen(
            [](Vector2D pos, Event::SCallbackInfo& info) {
                onMouseMove(pos, info);
            }
        );

    static auto mouseButton =
        Event::bus()->m_events.input.mouse.button.listen(
            [](IPointer::SButtonEvent event, Event::SCallbackInfo& info) {
                onMouseButton(event, info);
            }
        );

    static auto mouseAxis =
        Event::bus()->m_events.input.mouse.axis.listen(
            [](IPointer::SAxisEvent event, Event::SCallbackInfo& info) {
                onMouseAxis(event, info);
            }
        );

    static auto keyboardKey =
        Event::bus()->m_events.input.keyboard.key.listen(
            [](IKeyboard::SKeyEvent event, Event::SCallbackInfo& info) {
                onKeyboardKey(event, info);
            }
        );

    (void)renderPre;
    (void)renderStage;
    (void)mouseMove;
    (void)mouseButton;
    (void)mouseAxis;
    (void)keyboardKey;

    if (!HyprlandAPI::reloadConfig()) {
        notify(
            "[hypr3d] reloadConfig failed",
            CHyprColor{1.0f, 0.6f, 0.2f, 1.0f}
        );
    }

    return {
        "hypr3d",
        "A new perspective on window management",
        "Samine825",
        "0.3.0"
    };
}

APICALL EXPORT void PLUGIN_EXIT() {
    g_active = false;
    stopFramePump();
    g_transition = 0.0f;
    g_transitionTarget = 0.0f;

    if (Compat::setCursorHidden(false) && g_pHyprRenderer)
        g_pHyprRenderer->setCursorFromName("default", true);

    clearAimFocus();

    g_world.clear();
    g_renderWindows.clear();
    g_capture.releaseAll();

    // Put the windows back under the layout before the plugin goes away.
    unghostWindows();

    Compat::setPointerCapture(false);
    Compat::removePointerHook();

    g_monitor = nullptr;

    if (Render::GL::g_pHyprOpenGL) {
        Render::GL::g_pHyprOpenGL->makeEGLCurrent();
        g_scene.shutdown();
    }
}

} // namespace H3D
