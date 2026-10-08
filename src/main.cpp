#include <hyprland/src/plugins/PluginAPI.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/PlaneShape.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Collision/RayCast.h>

#include <hyprland/src/event/EventBus.hpp>
#include <hyprland/src/render/OpenGL.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/render/pass/PassElement.hpp>
#include <hyprland/src/render/gl/GLFramebuffer.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/desktop/view/Window.hpp>
#include <csignal>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <wayland-server-core.h>
#include <unistd.h>
#include <hyprland/src/Compositor.hpp>
#include <hyprland/src/managers/ANRManager.hpp>
#include <hyprland/src/helpers/AsyncDialogBox.hpp>
#include <hyprland/src/managers/EventManager.hpp>
#include <hyprland/src/managers/SeatManager.hpp>
#include <hyprland/src/managers/SessionLockManager.hpp>
#include <hyprland/src/config/supplementary/executor/Executor.hpp>
#include <hyprland/src/config/shared/actions/ConfigActions.hpp>
#include <hyprland/src/managers/eventLoop/EventLoopManager.hpp>
#include <hyprland/src/managers/fullscreen/FullscreenController.hpp>
#include <hyprland/src/config/values/ConfigValues.hpp>
#include <hyprland/src/desktop/state/WindowState.hpp>
#include <hyprland/src/config/ConfigValue.hpp>
#include <hyprland/src/devices/IKeyboard.hpp>
#include <hyprland/src/devices/IPointer.hpp>
#include <hyprland/src/pointer/PointerManager.hpp>
#include <hyprland/src/pointer/PointerController.hpp>
#include <hyprland/src/state/MonitorState.hpp>
#include <hyprutils/memory/UniquePtr.hpp>

// This system's lua headers (5.5) lost the extern "C" guard: including them
// from C++ mangles the API names and the plugin fails to load with
// "undefined symbol: _Z8lua_typeP9lua_Statei". liblua exports plain C
// symbols, so the linkage is forced here.
extern "C" {
#include <lua.h>
#include <lauxlib.h>
}

#include <linux/input-event-codes.h>
#include <turbojpeg.h>
#include <xkbcommon/xkbcommon.h>

#include "Render/GLScene.hpp"
#include "Input/AimFocus.hpp"
#include "Input/InputController.hpp"
#include "World/MapCollision.hpp"
#include "World/World3D.hpp"
#include "HyprlandCompat/WindowsCompat.hpp"
#include "HyprlandCompat/FocusCompat.hpp"
#include "HyprlandCompat/WindowCapture.hpp"
#include "HyprlandCompat/PointerHook.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace H3D {

static HANDLE PHANDLE = nullptr;

static GLScene         g_scene;
static InputController g_input;
static World3D::CWorld g_world;
static AimFocus        g_aim;

// The room as it was left: where the player stood and looked, the view (F5),
// and where every window and layer stood. Entering used to start at the spawn
// point with every window back on the wall each time (the Larch owner,
// 2026-10-06). Kept for the compositor's lifetime; hl.plugin.hypr3d.reset()
// forgets it.
struct SRememberedPose {
    // A weak reference proves it is the same surface: a closed window's locks
    // to nothing, so a new one at a reused address does not inherit its place.
    PHLWINDOWREF window;
    PHLLSREF     layer;
    Vec3         center{};
    float        yaw = 0.0f, pitch = 0.0f, roll = 0.0f;
};
struct SRoomMemory {
    bool  valid = false;
    Vec3  eye{};
    float yaw = 0.0f, pitch = 0.0f;
    int   viewMode = 0;
    std::unordered_map<std::uintptr_t, SRememberedPose> poses;
};
static SRoomMemory g_room;
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
static std::string g_lastRenderGate;  // why the 3D pass was skipped last frame
static double g_msUpdate3D = 0.0;     // per-section profiling (exponential avg)
static double g_msJolt     = 0.0;
static double g_msRender   = 0.0;
static double g_msRenderSpan = 0.0;    // one neighbour's pass, the same average
static std::string g_lastError;        // last caught handler exception

static void dumpErrorNow(const std::string& what) {
    g_lastError = what;
    std::ofstream out("/tmp/hypr3d-status.txt", std::ios::app);
    out << "!!! EXCEPTION " << what << "\n";
}
static bool g_reportedPointerHookError = false;

// Layout snapshot taken on entry; empty once the windows are back under the
// layout's control.
static std::vector<Compat::SWindowLayoutSave> g_layoutSaves;
static bool g_ghosted = false;

// What to draw this frame, rebuilt once per frame from the world.
static std::vector<GLScene::WindowRender> g_renderWindows;

// Per-window depth-slab silhouette (analytic rounded rectangle), rebuilt
// when the captured box or the reported corner radius changes. Cleared with
// the frame -- entries for closed windows die with it.
struct SWinOutline {
    int w = 0, h = 0, r = -1;
    std::shared_ptr<const std::vector<SOutlineLoop>> loops;
};
static std::unordered_map<std::uintptr_t, SWinOutline> g_winOutlines;

// Set when a resize gesture ends: the next capture pass makes one FORCED
// snapshot of that window, so the wall silhouette runs its full-resolution
// "box settled" refresh -- an idle window would otherwise never snapshot
// again and keep the last mid-drag (reduced-resolution) outline.
static std::uintptr_t g_skirtFinalRefreshId = 0;

// The window that currently owns keyboard focus, as the aim logic last set it.
static std::uintptr_t g_lastFocusId = 0;
// The window the focus lock holds (0 = none): it keeps the keyboard while the
// crosshair looks elsewhere. A GameMaker game under Proton takes no controller
// input without focus (Deltarune, measured 2026-10-07), so a game played with
// a pad in the room would stop at every glance away. Kept across leaving and
// entering the room; released by F6, by focus_lock(false), or when its window
// is gone.
static std::uintptr_t g_focusLockId = 0;

// 3D FPS-style mouse gestures. Super is the modifier: LMB moves the aimed
// window in the 3D room, RMB resizes its real Hyprland/Wayland geometry.
enum class EPointerGesture : uint8_t {
    None,
    Move3D,
    ResizeReal,
    WheelRoll,
    MapDrag // carrying a dynamic (static = false) scene object
};

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
    // Crosshair position on the real window at grab time, in surface px. The
    // resize is driven by the aim's DELTA from this point, so the grabbed
    // corner never snaps to the crosshair when the gesture starts.
    Vector2D      grabPx{};
    int           edgeX = 0; // -1 left, +1 right
    int           edgeY = 0; // +1 top, -1 bottom
};

// Player spawn point in the room, set via hl.plugin.hypr3d.config().
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

// Super+wheel-PRESS roll: rotating the aimed window around its normal by
// sweeping the crosshair around the window center. The swept angle is
// measured in WORLD space around the normal, which is invariant to the roll
// itself -- only camera rotation drives it, so the gesture never feeds back
// into itself.
static struct SWheelRotate {
    bool           active     = false;
    bool           model      = false;  // rolling a scene object, not a window
    size_t         modelIndex = SIZE_MAX;
    std::uintptr_t id         = 0;
    Vec3           center     = {};
    Vec3           normal     = {};
    Vec3           reference  = {};  // crosshair point - center at grab
    float          startRoll  = 0.0f;
} s_wheelRot;

// Super+wheel hover zoom: glides the aimed window along the ray from the
// camera through the window toward the target distance (exponential lerp,
// same feel as the drag zoom). Cleared when a drag takes over the window or
// the entity disappears.
static std::uintptr_t s_zoomId     = 0;
static Vec3           s_zoomDir    = {};
static Vec3           s_zoomAnchor = {}; // camera position at the last wheel event
static double         s_zoomCur    = 0.0;
static double         s_zoomTarget = 0.0;

// Fullscreen passthrough: when Hyprland fullscreens a window while the 3D
// view is open, the window's quad animates to face the (frozen) camera and
// stretch exactly over the frustum, then the plugin hands the screen back to
// the 2D compositor: input is released, the cursor reappears, the 3D scene
// stops rendering. Exiting fullscreen reverses it -- the quad is back, input
// captured, cursor hidden. The real window is forced to the monitor box
// during 2D so the compositor shows it truly fullscreened.
static void resetPointerGesture();

enum class EFullscreenPhase : uint8_t { None, To2D, In2D, To3D };
static EFullscreenPhase g_fsPhase = EFullscreenPhase::None;
static PHLWINDOWREF     g_fsWindow;
static CBox             g_fsRestoreBox{}; // floating box before the fullscreen
static CBox             g_fsMonitorBox{}; // the fullscreen target box
static float            g_fsDistance = 8.0f; // fov-derived, set by startTo2D
static Vec3             g_fsStartCenter{}, g_fsEndCenter{};
static float g_fsStartYaw = 0.f, g_fsStartPitch = 0.f;
static float g_fsEndYaw = 0.f, g_fsEndPitch = 0.f;
static float g_fsStartRoll = 0.f;
static std::chrono::steady_clock::time_point g_fsPhaseStart{};
static bool  g_fsWasOn = false; // a fullscreen window existed since the last poll
static PHLWINDOWREF g_fsLastFSWindow; // the most recent fullscreen window
// A window that went fullscreen on a monitor other than the one in front of
// the player, and the workspace it came from: brought in front for the
// fullscreen, sent home when it ends. The fullscreen windows on the other
// monitors at the last poll, so only a NEW one is brought (a game already
// fullscreen there when the room opened stays where it is).
static PHLWINDOWREF    g_fsHomeWindow;
static PHLWORKSPACEREF g_fsHomeWorkspace;
static std::vector<std::uintptr_t> g_fsElsewhere;
static bool g_fsElsewhereKnown = false;
static std::uintptr_t g_fsCurrentId = 0; // the id of the CURRENT fullscreen window, 0 = none
static float g_fsAlpha = 1.0f;  // composite alpha during the transition

// The room's own fade for the NON-fullscreen windows: 1 = visible. Driven by
// the pump (deterministic ticks), sampled into WindowRender::alpha -- the
// snapshot alpha is baked at capture time and never advances (the buffer
// does not change during an alpha fade), so a snapshot-driven fade freezes
// after a couple of frames. Hyprland's own fade animates the real channels;
// ours guarantees the room matches it.
static float g_fsFade = 1.0f;
static std::chrono::steady_clock::time_point g_fsFadeLast{};
static float g_fsRawP   = 0.0f;  // raw (pre-smoothstep) transition progress
static float g_fsSavedRoll   = 0.0f; // the window's roll before the fullscreen
static float g_fsRollAtStart = 0.0f; // roll at the To3D start (aborts may differ)
static int   g_fsAssertFrames = 0;   // re-assert the restored box...
static CBox  g_fsAssertBox{};
static int   g_fsStableCount  = 0;   // consecutive frames the box matched

// Last known real box of every window OUTSIDE any fullscreen transition,
// refreshed each frame. The pre-fullscreen box must come from here: by the
// time the pump detects the fullscreen, Hyprland has already resized the
// window to the monitor, so the live geometry is NOT the pre-FS box.
static std::unordered_map<std::uintptr_t, CBox> g_fsStableBoxes;
static bool  g_captureReleasePending = false;
static constexpr float kFsAnimDuration = 0.6f;

static void pollFullscreen();
static void startTo2D(const PHLWINDOW& window, bool captureRestoreBox = true);

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

// Plugin settings, set from lua via hl.plugin.hypr3d.config({...}). Missing
// keys keep their current value, so a partial config only touches what it
// names; wrong-typed keys raise a lua error. No value ranges: what you
// write is what the plugin uses.
// --- world ------------------------------------------------------------------
static std::string g_cfgPanorama;                // panorama image path
static std::string g_cfgMonitor;                 // monitor name (e.g. "DP-1"), empty = focused
static bool        g_cfgGrid = true;             // base grid platform on/off
static bool        g_cfgShadows = true;          // shadows under the windows (updateShadows)
static bool        g_cfgHud = true;              // the rice's bar and notifications in view (g_hudItems)
static bool        g_cfgSpan = true;             // the room across every monitor

// --- windows ----------------------------------------------------------------
static float       g_cfgWindowScale   = 0.5f;    // room multiplier on window size
static float       g_cfgSpawnDistance = 5.0f;    // units in front of the camera
static float       g_cfgWindowDepth   = 0.05f;   // slab thickness, 0 = flat quads

// --- player -----------------------------------------------------------------
static float       g_cfgLookInertia   = 0.03f;   // seconds, 0 = off
static float       g_cfgMoveInertia   = 0.18f;   // flight glide, seconds, 0 = off
static float       g_cfgGravity       = 14.0f;   // world.gravity, m/s^2 down
static float       g_cfgMoveSpeed     = 4.0f;    // world units / second
static float       g_cfgSensitivity   = 0.0025f; // radians per pointer count
static bool        g_playerFlying     = true;    // false = walk / jump / gravity
static bool        g_playerCollision  = true;    // reserved: the capsule is
        // the movement system, so this only gates future per-object contact
static bool        g_cfgWalkBob       = true;    // view-only walk bob (walking only)
static GLScene::SPlayerCfg g_playerCfg;          // the player character
static float        g_playerAnimSpeed[4] = {1.f, 1.f, 1.f, 1.f};

// Feet position; eyes ride kEyeHeight above (spawn 0,0,0 = standing on
// the grid platform at world zero).
static Vec3 g_playerSpawn{0.0f, 0.0f, 0.0f};

// Walking physics state (grounded comes from the Jolt body's ground ray).
static bool g_grounded = false;

// --- scene: unlimited named glTF objects ------------------------------------
struct SSceneObjectCfg {
    std::string name; // the lua table key ("scene.<name>")
    std::string path;
    Vec3        position{}, rotationDeg{}, scale{1.0f, 1.0f, 1.0f};
    float       emissiveScale = 1.0f;
    bool        flat = false; // false = headlight half-lambert shading
    bool        collision = true;
    bool        dynamic = false; // static = false -> grabbable with Super+LMB
    bool        physics = false; // gravity + world collisions (dynamic only)
    CMapModel::ECenter center = CMapModel::ECenter::Logical;
    Vec3        centerOffset{}; // extra pivot shift, local units
};

static std::vector<SSceneObjectCfg> g_sceneObjects;
static std::vector<SSceneObjectCfg> g_sceneLuaState; // the last parsed config
static bool g_sceneLuaValid = false;


// Index of the currently grabbed dynamic object, SIZE_MAX when none.
static size_t g_mapGrabIndex = SIZE_MAX;

// Per-object physics (scene object physics = true): fall velocity + the AABB
// cache (recomputed when the object's generation changes).
struct SObjPhys {
    Vec3 vel{};
};
static std::vector<SObjPhys>   g_objPhys;
static std::vector<Vec3>       g_objAABBLo, g_objAABBHi;
static std::vector<uint32_t>   g_objAABBGens;

// Super+wheel hover zoom for a dynamic scene object (the model-class twin of
// the window hover zoom): the aimed object's center rides the zoomed
// distance on the camera-object line.
static size_t s_modelZoomId   = SIZE_MAX;
static float  s_modelZoomDist = 0.0f;
static float  g_mapGrabDist  = 0.0f; // camera-to-object-CENTER distance


// F3 debug HUD: collision wireframe + room info overlay.
static bool        g_debugHud = false;
static float       g_debugFps = 0.0f;

// C-key view zoom: g_zoomLevel glides toward the target (the wheel-adjusted
// magnification while C is held, 1x when released). The wheel level resets
// to kZoomBase on every press -- it does not survive the key release.
static bool  g_zoomHeld  = false;
static float g_zoomWheel = 2.0f;
static float g_zoomLevel = 1.0f;
static constexpr float kZoomBase = 2.0f;

// F5 view modes: 0 first person, 1 third person behind, 2 third person front.
static int g_viewMode = 0;
static float g_playerCamDist = 0.f; // smoothed third-person distance
static constexpr float kThirdDist = 2.5f;

// Per-object collision trees (see update3D): one small BVH per scene
// object, used by picking (modelRayHit). The static map's tree builds once;
// the grid platform slab lives in Jolt (syncFloorBody).
static std::vector<CMapCollision> g_objTrees;
static std::vector<uint32_t>    g_objTreeGens;

// --- Jolt Physics: rigid body simulation for scene objects ------------------
// Object layers: static geometry vs moving bodies. Moving bodies collide
// with everything; static-vs-static never (they cannot move anyway).
constexpr JPH::ObjectLayer LAYER_STATIC = 0;
constexpr JPH::ObjectLayer LAYER_MOVING = 1;
// A carried object is pose-driven: it sits in NO contact pair at all, so it
// can neither shove the player body around nor fight its own teleport.
constexpr JPH::ObjectLayer LAYER_GRABBED = 2;
constexpr JPH::ObjectLayer NUM_OBJECT_LAYERS = 3;

constexpr JPH::BroadPhaseLayer BP_LAYER_STATIC(0);
constexpr JPH::BroadPhaseLayer BP_LAYER_MOVING(1);

class SBroadPhaseLayerInterface final : public JPH::BroadPhaseLayerInterface {
  public:
    SBroadPhaseLayerInterface() {
        mObjectToBroadPhase[LAYER_STATIC] = BP_LAYER_STATIC;
        m_objectToBroadPhase2[LAYER_MOVING] = BP_LAYER_MOVING;
        m_objectToBroadPhase2[LAYER_GRABBED] = BP_LAYER_MOVING;
    }

    JPH::uint GetNumBroadPhaseLayers() const override {
        return 2;
    }

    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
        return layer == LAYER_STATIC ? BP_LAYER_STATIC : BP_LAYER_MOVING;
    }

  private:
    JPH::BroadPhaseLayer m_objectToBroadPhase2[NUM_OBJECT_LAYERS];
    JPH::BroadPhaseLayer mObjectToBroadPhase[NUM_OBJECT_LAYERS] = {};
};

class SObjectVsBroadPhase final : public JPH::ObjectVsBroadPhaseLayerFilter {
  public:
    bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer bp) const override {
        if (layer == LAYER_STATIC)
            return bp == BP_LAYER_MOVING;
        return true; // moving collides with everything
    }
};

class SObjectLayerPair final : public JPH::ObjectLayerPairFilter {
  public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        if (a == LAYER_GRABBED || b == LAYER_GRABBED)
            return false; // carried: pose-driven, collides with nothing
        if (a == LAYER_STATIC)
            return b == LAYER_MOVING;
        return true; // moving vs moving and moving vs static
    }
};

static SBroadPhaseLayerInterface g_bpLayers;
static SObjectVsBroadPhase      g_objVsBp;
static SObjectLayerPair         g_objPair;

static JPH::PhysicsSystem*  g_joltSystem = nullptr;
static JPH::BodyInterface*  g_bodyIf     = nullptr;

// Per-scene-object Jolt body state (index-aligned with g_sceneObjects).
struct SObjJolt {
    JPH::BodyID body{};
    uint32_t    meshGen = 0; // model generation the body was built from
    Vec3        builtScale{1.0f, 1.0f, 1.0f}; // body-space mapping used at
    Vec3        builtPivot{};                 // build time (scale + pivot)
    bool        valid = false;
};
static std::vector<SObjJolt> g_joltBodies;

// A static object's mesh shape, cooked off the main thread: Jolt building
// the MeshShape of a 95 000-triangle model held Hyprland ~180 ms (-O2,
// perf). Jolt is made for this -- MeshShape creation was tuned for many
// threads building meshes at once.
struct SShapeJob {
    uint32_t             meshGen = 0; // what the shape is built from
    Vec3                 scale{}, pivot{};
    JPH::Ref<JPH::Shape> shape;       // empty when Jolt refused the mesh
    std::atomic<bool>    done{false};
    std::jthread         thread;      // last: joined before the rest goes
};
static std::vector<std::unique_ptr<SShapeJob>> g_shapeJobs; // by object

// A static object's first shape is still cooking: there is nothing to stand
// on yet where it will be.
static bool joltShapesPending() {
    for (size_t i = 0; i < g_shapeJobs.size() && i < g_joltBodies.size(); ++i)
        if (g_shapeJobs[i] && !g_joltBodies[i].valid &&
            !g_shapeJobs[i]->done.load(std::memory_order_acquire))
            return true;
    return false;
}

// Gravity factor saved while a wheel-roll gesture suspends the object's
// simulation (captured ONCE at grab -- re-reading it per frame would store
// the suspended zero and never restore it).
static float g_rolledGravity = 1.0f;

// Creates/removes/teleports the Jolt body of each scene object so it
// matches the config state. Body classes:
//   static = true              -> Static body (the location's geometry)
//   dynamic && physics         -> Dynamic body (the simulation owns it)
//   dynamic && !physics        -> Kinematic body (grabbable, solid, inert)
//   collision = false          -> no body (no occlusion, no collision)


static bool joltInit() {
    static bool done = false;
    if (done)
        return true;

    JPH::RegisterDefaultAllocator();
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();

    g_joltSystem = new JPH::PhysicsSystem();
    g_joltSystem->Init(512, 0, 1024, 1024, g_bpLayers, g_objVsBp, g_objPair);
    // Same gravity as the player's walk physics.
    g_joltSystem->SetGravity(JPH::Vec3(0.f, -g_cfgGravity, 0.f));

    g_bodyIf = &g_joltSystem->GetBodyInterface();
    done = true;
    return true;
}

static void joltShutdown() {
    // A shape still cooking finishes first; Jolt cannot stop mid-build.
    g_shapeJobs.clear();

    if (g_joltSystem) {
        delete g_joltSystem;
        g_joltSystem = nullptr;
    }
    if (JPH::Factory::sInstance) {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
}

// --- the player's own physics body ------------------------------------------
// A dynamic Jolt capsule driven by per-frame velocity control: the keys own
// the horizontal axes (the inertia glide computes s_moveVel), Jolt owns
// gravity, contacts and the pose. Active physics bodies meet it through the
// exact same contact pipeline they use between themselves -- it can push
// them, stand on them, be stood on. Rotation is locked out with
// translation-only DOFs: the solver keeps the capsule upright, and the look
// direction stays in our own yaw/pitch angles.
static JPH::BodyID g_playerBody{};      // invalid until joltInit succeeds
static JPH::BodyID g_floorBody{};       // grid platform slab (world.grid)
static bool        g_playerJumpQueued = false;

// Body center -> eye offset: the eye rides kEyeHeight above the feet.
static constexpr float PLAYER_EYE_OFF =
    Camera::kEyeHeight - Camera::kBodyHeight * 0.5f;

// A 0.6x1.8 capsule driven by velocity: the player's body and the
// companion's.
static JPH::BodyID capsuleBody(const JPH::RVec3& center) {
    JPH::CapsuleShapeSettings CAPSULE(
        Camera::kBodyHeight * 0.5f - Camera::kBodyHalfWidth, // cylinder half
        Camera::kBodyHalfWidth);                             // -> 0.6x1.8
    CAPSULE.SetEmbedded();
    auto RES = CAPSULE.Create();
    if (RES.HasError())
        return {};

    JPH::BodyCreationSettings BCS(
        RES.Get(), center, JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic,
        LAYER_MOVING);
    BCS.mAllowedDOFs   = JPH::EAllowedDOFs::TranslationX |
                         JPH::EAllowedDOFs::TranslationY |
                         JPH::EAllowedDOFs::TranslationZ;
    BCS.mMotionQuality = JPH::EMotionQuality::LinearCast; // cast the step:
                          // never tunnels the map's zero-thickness sheets
    // Friction ZERO. The commanded into-wall velocity makes the solver
    // cancel ~5 m/s every sub-step -- a normal force on the order of
    // 450*m -- and any friction then lets the wall hold the player against
    // gravity (measured: a -0.16 m/s creep instead of free fall) and
    // stick-slip while sliding (the wall-press shaking). The player is
    // velocity-driven: friction only ever hurt.
    BCS.mFriction      = 0.0f;
    BCS.mRestitution   = 0.0f;
    BCS.mAllowSleeping = false; // always controlled
    BCS.mGravityFactor = 0.0f;  // set per-frame by the movement mode

    auto* B = g_bodyIf->CreateBody(BCS);
    if (!B)
        return {};
    g_bodyIf->AddBody(B->GetID(), JPH::EActivation::Activate);
    return B->GetID();
}

static void ensurePlayerBody() {
    if (!g_joltSystem || !g_playerBody.IsInvalid())
        return;

    // Spawned under the camera (the camera was placed by the spawn reset).
    const auto& CAM = g_scene.camera();
    g_playerBody = capsuleBody(JPH::RVec3(
        CAM.position.x, CAM.position.y - PLAYER_EYE_OFF, CAM.position.z));
}

// The grid platform as real physics: a slab matching the visible grid
// extent (the lines run -20..20), top at world zero. Both the player and
// physics objects land on it.
static void syncFloorBody() {
    if (!g_joltSystem)
        return;

    if (g_cfgGrid && g_floorBody.IsInvalid()) {
        JPH::BoxShapeSettings SLAB(JPH::Vec3(20.0f, 0.5f, 20.0f));
        SLAB.SetEmbedded();
        auto RES = SLAB.Create();
        if (RES.HasError())
            return;

        JPH::BodyCreationSettings BCS(
            RES.Get(), JPH::RVec3(0.f, -0.5f, 0.f), JPH::Quat::sIdentity(),
            JPH::EMotionType::Static, LAYER_STATIC);
        auto* B = g_bodyIf->CreateBody(BCS);
        if (!B)
            return;
        g_bodyIf->AddBody(B->GetID(), JPH::EActivation::DontActivate);
        g_floorBody = B->GetID();
    } else if (!g_cfgGrid && !g_floorBody.IsInvalid()) {
        g_bodyIf->RemoveBody(g_floorBody);
        g_bodyIf->DestroyBody(g_floorBody);
        g_floorBody = JPH::BodyID();
    }
}

// Ground probe: a short ray straight down from the body center (feet with
// 0.15 slack). Back-face culling skips the capsule's own underside.
static bool playerGrounded() {
    if (!g_joltSystem || g_playerBody.IsInvalid())
        return false;

    const auto POS = g_bodyIf->GetPosition(g_playerBody);
    const JPH::RRayCast RAY{
        POS, JPH::Vec3(0.f, -(Camera::kBodyHeight * 0.5f + 0.15f), 0.f)};
    JPH::RayCastResult HIT;
    // The ray STARTS INSIDE the player capsule: unfiltered, the cast reports
    // the player's own shape at fraction 0 and grounded is true for the
    // whole flight (flappy-bird jumps). Exclude self.
    const JPH::IgnoreSingleBodyFilter SKIP_SELF(g_playerBody);
    return g_joltSystem->GetNarrowPhaseQuery().CastRay(
        RAY, HIT, JPH::BroadPhaseLayerFilter(), JPH::ObjectLayerFilter(),
        SKIP_SELF);
}

// --- the companion: a second avatar, driven from outside --------------------
// hl.plugin.hypr3d.companion(verb, target) is its whole interface. A separate
// bridge process speaks for it to an AI over the Neuro SDK protocol, so no
// network code lives in the compositor. Its rights are its own body and its
// own gaze: it walks a straight line to a target and turns to face one; it
// types into nothing and never walks into the player. The body is a second
// capsule like the player's and lives as long as the plugin, so leaving the
// room keeps it where it stood. What it does goes out on socket2 as
// hypr3d>>{json}, stamped with CLOCK_MONOTONIC.
enum class ECompanionMode : uint8_t { Idle, Walk, Turn };
enum class ECompanionTarget : uint8_t { Player, Spawn, Window };

struct SCompanion {
    JPH::BodyID      body{};
    float            yaw  = 0.f; // where it faces, the camera's convention
    ECompanionMode   mode = ECompanionMode::Idle;
    ECompanionTarget kind = ECompanionTarget::Spawn;
    std::string      target;     // the name it was sent to
    std::uintptr_t   windowId = 0;
    Vec3             start{};    // where the walk began
    bool             stepped   = false;
    float            stuck     = 0.f; // seconds walking without getting anywhere
    float            lastSpeed = 0.f; // commanded on the previous frame
};
static SCompanion g_companion;
// It appears on the first call (a bridge asks for the state when it
// starts), not in every room: without an AI it would only stand there.
static bool g_companionCalled = false;

// companion("see"): one picture from its eye at a time, at most one a
// second (agreed with the AI side, 2026-10-07). Asked here, handed to the
// scene by updateCompanion where the body's place is known, delivered by
// deliverSight.
static bool      g_sightAsked   = false; // not yet handed to the scene
static bool      g_sightPending = false; // asked and not yet delivered
static long long g_sightAskedNs = 0;
static long long g_sightTakenNs = 0;     // the frame that drew it

static constexpr float kCompanionSpeed = 3.0f;  // m/s; the player flies at 4
static constexpr float kCompanionTurn  = 8.0f;  // rad/s
static constexpr float kNearPlayer     = 1.5f;  // go_to "player" ends here
static constexpr float kNearPoint      = 0.3f;  // every other target
static constexpr float kWindowStandOff = 1.5f;  // in front of a window
static constexpr float kPersonalSpace  = 0.9f;  // body centers, never closer
static constexpr float kStuckSeconds   = 1.5f;
static constexpr float kFloorHalf      = 20.0f; // the slab in syncFloorBody

// Windows are named by their app class in lower case, numbered from the
// second of a class on ("foot", "foot 2"). A number stays with its window
// while it lives; the next window of the class gets the lowest free one.
struct SCompanionName {
    PHLWINDOWREF window;
    std::string  base;
    int          number = 1;

    std::string name() const {
        return number == 1 ? base : base + " " + std::to_string(number);
    }
};
static std::unordered_map<std::uintptr_t, SCompanionName> g_companionNames;

static long long monotonicNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

static std::string jsonString(const std::string& s) {
    std::string out = "\"";
    for (const unsigned char C : s) {
        if (C == '"' || C == '\\') {
            out += '\\';
            out += static_cast<char>(C);
        } else if (C < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", C);
            out += buf;
        } else
            out += static_cast<char>(C);
    }
    return out + "\"";
}

static void postRoomEvent(const std::string& json) {
    if (g_pEventManager)
        g_pEventManager->postEvent(SHyprIPCEvent{"hypr3d", json});
}

// A locked session (ext-session-lock) is the lock screen's alone: the room
// draws nothing and takes no input. It used to draw over the lock surface at
// RENDER_LAST_MOMENT -- every window, live, and the lock screen unseen
// (measured 2026-10-07 in the caelestia sandbox, whose lock is a
// WlSessionLock). Typed keys went to the lock there, not to the windows.
static bool sessionLocked() {
    return g_pSessionLockManager && g_pSessionLockManager->isSessionLocked();
}

// Open = on screen and taking commands: not while it fades out, not while a
// fullscreen window has it paused.
static bool roomOpen() {
    return g_active && g_transitionTarget > 0.5f &&
        g_fsPhase != EFullscreenPhase::In2D && !sessionLocked();
}

// The windows in the room with their names, sorted by name. Layers (bars,
// notifications) are no targets.
static std::vector<std::pair<std::uintptr_t, std::string>> companionWindows() {
    std::erase_if(g_companionNames,
                  [](const auto& KV) { return !KV.second.window.lock(); });

    std::vector<std::uintptr_t> ids;
    for (const auto& E : g_world.entities()) {
        const auto W = Compat::findWindowById(E.id);
        if (!W)
            continue;

        if (!g_companionNames.contains(E.id)) {
            // Only [a-z0-9._-]: a name is always safe to pass back in.
            std::string base;
            for (unsigned char C : W->m_class.empty() ? W->m_initialClass : W->m_class) {
                if (C >= 'A' && C <= 'Z')
                    C = C - 'A' + 'a';
                const bool KEEP = (C >= 'a' && C <= 'z') || (C >= '0' && C <= '9') ||
                    C == '.' || C == '-' || C == '_';
                base += KEEP ? static_cast<char>(C) : '-';
            }
            if (base.empty())
                base = "window";
            if (base.size() > 40)
                base.resize(40);

            // "player" and "spawn" are taken by the other targets.
            int number = base == "player" || base == "spawn" ? 2 : 1;
            for (bool taken = true; taken;) {
                taken = false;
                for (const auto& [ID, N] : g_companionNames)
                    if (N.base == base && N.number == number) {
                        taken = true;
                        ++number;
                        break;
                    }
            }
            g_companionNames.emplace(E.id, SCompanionName{W, base, number});
        }
        ids.push_back(E.id);
    }

    std::ranges::sort(ids, [](std::uintptr_t a, std::uintptr_t b) {
        const auto& A = g_companionNames.at(a);
        const auto& B = g_companionNames.at(b);
        return A.base != B.base ? A.base < B.base : A.number < B.number;
    });

    std::vector<std::pair<std::uintptr_t, std::string>> out;
    for (const auto ID : ids)
        out.emplace_back(ID, g_companionNames.at(ID).name());
    return out;
}

static std::string companionTargetList() {
    std::string out = "player, spawn";
    for (const auto& [ID, NAME] : companionWindows())
        out += ", " + NAME;
    return out;
}

static bool companionResolve(const std::string& name, ECompanionTarget& kind,
                             std::uintptr_t& id) {
    if (name == "player" || name == "spawn") {
        kind = name == "player" ? ECompanionTarget::Player : ECompanionTarget::Spawn;
        return true;
    }
    for (const auto& [ID, NAME] : companionWindows())
        if (NAME == name) {
            kind = ECompanionTarget::Window;
            id   = ID;
            return true;
        }
    return false;
}

// Where a walk to the target ends this frame, what to face there and how
// close counts as there. False with the reason when the target is gone.
static bool companionGoal(ECompanionTarget kind, std::uintptr_t id,
                          const std::string& name, const Vec3& at, Vec3& point,
                          std::optional<Vec3>& face, float& near,
                          std::string& why) {
    face.reset();
    near = kNearPoint;

    switch (kind) {
        case ECompanionTarget::Player: {
            if (g_playerBody.IsInvalid()) {
                why = "The player has no body";
                return false;
            }
            const auto P = g_bodyIf->GetPosition(g_playerBody);
            point = Vec3{P.GetX(), 0.f, P.GetZ()};
            face  = point;
            near  = kNearPlayer;
            return true;
        }
        case ECompanionTarget::Spawn: point = Vec3{g_playerSpawn.x, 0.f, g_playerSpawn.z}; break;
        case ECompanionTarget::Window: {
            const auto* E = g_world.find(id);
            if (!E) {
                why = "'" + name + "' " +
                    (Compat::findWindowById(id) ? "left the room" : "was closed");
                return false;
            }

            // On the floor along the window's normal, the side its content
            // faces; a window lying flat is approached from where the
            // companion stands.
            const Vec3 N = g_world.normalOf(id);
            float hx = N.x, hz = N.z, len = std::sqrt(hx * hx + hz * hz);
            if (len < 0.2f) {
                hx  = at.x - E->center.x;
                hz  = at.z - E->center.z;
                len = std::sqrt(hx * hx + hz * hz);
                if (len < 1e-3f)
                    hx = 0.f, hz = 1.f, len = 1.f;
            }
            point = Vec3{E->center.x + hx / len * kWindowStandOff, 0.f,
                         E->center.z + hz / len * kWindowStandOff};
            face  = Vec3{E->center.x, 0.f, E->center.z};
            break;
        }
    }
    return true;
}

// A walk ends on the floor. The player is the exception: it may fly past
// the edge, and the companion then stops there (updateCompanion).
static bool companionReachable(ECompanionTarget kind, const std::string& name,
                               const Vec3& point, std::string& why) {
    const float EDGE = kFloorHalf - Camera::kBodyHalfWidth;
    if (kind == ECompanionTarget::Player ||
        (std::fabs(point.x) <= EDGE && std::fabs(point.z) <= EDGE))
        return true;
    why = kind == ECompanionTarget::Window ?
        "The point in front of '" + name + "' is off the floor" :
        "'" + name + "' is off the floor";
    return false;
}

static void companionEvent(const char* state, const std::string& reason = {}) {
    std::string json = std::string("{\"event\":\"companion\",\"state\":\"") + state +
        "\",\"target\":" + jsonString(g_companion.target);
    if (!reason.empty())
        json += ",\"reason\":" + jsonString(reason);
    postRoomEvent(json + ",\"t\":" + std::to_string(monotonicNs()) + "}");
}

// Stands still. A walk cut short says why, unless `state` is null (stop).
static void companionHalt(const char* state, const std::string& reason) {
    if (g_companion.mode == ECompanionMode::Walk && state)
        companionEvent(state, reason);

    g_companion.mode      = ECompanionMode::Idle;
    g_companion.stuck     = 0.f;
    g_companion.lastSpeed = 0.f;
    if (!g_companion.body.IsInvalid() && g_bodyIf) {
        const auto V = g_bodyIf->GetLinearVelocity(g_companion.body);
        g_bodyIf->SetLinearVelocity(g_companion.body, JPH::Vec3(0.f, V.GetY(), 0.f));
    }
}

// Turns toward `to` at the turn rate; true once it faces it.
static bool companionTurn(const Vec3& from, const Vec3& to, float dt) {
    const float DX = to.x - from.x, DZ = to.z - from.z;
    if (DX * DX + DZ * DZ < 1e-6f)
        return true;

    const float WANT = std::atan2(DX, -DZ);
    const float DIFF = std::remainder(WANT - g_companion.yaw, 2.f * std::numbers::pi_v<float>);
    const float STEP = kCompanionTurn * dt;
    if (std::fabs(DIFF) <= STEP) {
        g_companion.yaw = WANT;
        return true;
    }
    g_companion.yaw += DIFF > 0.f ? STEP : -STEP;
    return false;
}

// Its place when it is new and after a reset: a step ahead of the player's
// spawn and to the right, turned toward it. Feet, like the player's spawn.
static void placeCompanion() {
    const float YAW = std::atan2(-g_playerSpawn.x, g_playerSpawn.z); // as placeAtSpawn
    const Vec3 FEET{
        g_playerSpawn.x + std::sin(YAW) * 1.5f + std::cos(YAW) * 1.2f,
        g_playerSpawn.y,
        g_playerSpawn.z - std::cos(YAW) * 1.5f + std::sin(YAW) * 1.2f,
    };
    g_companion.yaw =
        std::atan2(g_playerSpawn.x - FEET.x, -(g_playerSpawn.z - FEET.z));

    if (g_companion.body.IsInvalid() || !g_bodyIf)
        return;
    g_bodyIf->SetPositionAndRotation(
        g_companion.body,
        JPH::RVec3(FEET.x, FEET.y + Camera::kBodyHeight * 0.5f, FEET.z),
        JPH::Quat::sIdentity(), JPH::EActivation::Activate);
    g_bodyIf->SetLinearVelocity(g_companion.body, JPH::Vec3::sZero());
}

static void ensureCompanionBody() {
    if (!g_joltSystem || !g_companion.body.IsInvalid())
        return;

    g_companion.body = capsuleBody(JPH::RVec3::sZero());
    placeCompanion();
}

static bool companionStart(bool walk, const std::string& name, std::string& why) {
    if (!roomOpen()) {
        why = "The room is closed";
        return false;
    }
    ensureCompanionBody();
    if (g_companion.body.IsInvalid()) {
        why = "The companion has no body: the physics did not start";
        return false;
    }
    if (walk && g_floorBody.IsInvalid()) {
        why = "There is no floor to walk on (world.grid is off)";
        return false;
    }

    ECompanionTarget kind = ECompanionTarget::Spawn;
    std::uintptr_t   id   = 0;
    if (!companionResolve(name, kind, id)) {
        why = "Unknown target '" + name + "'. Targets now: " + companionTargetList();
        return false;
    }

    const auto P = g_bodyIf->GetPosition(g_companion.body);
    const Vec3 AT{P.GetX(), P.GetY(), P.GetZ()};
    Vec3 point{};
    std::optional<Vec3> face;
    float near = 0.f;
    if (!companionGoal(kind, id, name, AT, point, face, near, why) ||
        (walk && !companionReachable(kind, name, point, why)))
        return false;

    // A walk under way is replaced without a word: the new command says
    // what became of it.
    g_companion.mode      = walk ? ECompanionMode::Walk : ECompanionMode::Turn;
    g_companion.kind      = kind;
    g_companion.windowId  = id;
    g_companion.target    = name;
    g_companion.start     = AT;
    g_companion.stepped   = false;
    g_companion.stuck     = 0.f;
    g_companion.lastSpeed = 0.f;
    return true;
}

// Portals (hl.plugin.hypr3d.portal): a picture standing in the room, and
// walking into it runs its command through Hyprland's own executor -- the
// one hl.exec_cmd and an exec keybind use, so a game started from a portal
// is started as from a key. A portal lives as long as the plugin.
struct SPortal {
    GLScene::SPortalSpec spec;
    std::string          command;
    bool                 inside = false; // the player stood in it last frame
    double               quietUntil = 0.0; // no second start before this
};
static std::vector<SPortal> g_portals;
static double nowSeconds();
static void   notify(const std::string& text, const CHyprColor& color);
static constexpr float kPortalFront = 2.5f; // m in front of the player, for front = true

static void syncPortals() {
    std::vector<GLScene::SPortalSpec> specs;
    for (const auto& P : g_portals)
        specs.push_back(P.spec);
    g_scene.setPortals(specs);
}

// The player's body centre: the capsule's, or under the eye without one.
static Vec3 playerCenter() {
    if (!g_playerBody.IsInvalid() && g_bodyIf) {
        const auto P = g_bodyIf->GetPosition(g_playerBody);
        return Vec3{P.GetX(), P.GetY(), P.GetZ()};
    }
    const auto& CAM = g_scene.camera();
    return CAM.position - Vec3{0.f, PLAYER_EYE_OFF, 0.f};
}

// Once per frame: a player who steps into a portal's middle -- within the
// inner 70 % of its width, within its height, a quarter metre either side of
// its plane -- starts its command, once per entry, not twice in 3 s.
static void updatePortals() {
    if (g_portals.empty() || !roomOpen())
        return;
    const Vec3 P = playerCenter();
    for (auto& PORTAL : g_portals) {
        const float ASPECT = g_scene.portalAspect(PORTAL.spec.name);
        if (ASPECT <= 0.0f)
            continue;
        const float W = PORTAL.spec.width, H = W * ASPECT;
        const Vec3  FACE{std::sin(PORTAL.spec.yaw), 0.f, -std::cos(PORTAL.spec.yaw)};
        const Vec3  RIGHT = normalize(cross(FACE * -1.0f, Vec3{0.f, 1.f, 0.f}));
        const Vec3  D = P - PORTAL.spec.base;
        const bool  INSIDE = std::fabs(dot(D, FACE)) < 0.25f && std::fabs(dot(D, RIGHT)) < W * 0.35f &&
            D.y >= 0.0f && D.y <= H;
        if (INSIDE && !PORTAL.inside && nowSeconds() >= PORTAL.quietUntil) {
            PORTAL.quietUntil = nowSeconds() + 3.0;
            const auto PID = Config::Supplementary::executor()->spawn(PORTAL.command);
            postRoomEvent("{\"event\":\"portal\",\"name\":" + jsonString(PORTAL.spec.name) +
                          ",\"started\":" + (PID ? "true" : "false") + ",\"t\":" +
                          std::to_string(monotonicNs()) + "}");
            notify(PID ? "[hypr3d] portal " + PORTAL.spec.name + ": starting" :
                         "[hypr3d] portal " + PORTAL.spec.name + ": its command did not start",
                   PID ? CHyprColor{0.2f, 0.8f, 0.4f, 1.0f} : CHyprColor{1.0f, 0.2f, 0.2f, 1.0f});
        }
        PORTAL.inside = INSIDE;
    }
}

// Once per frame after the physics step: steer for the next one.
static void updateCompanion(float dt) {
    if (!g_companionCalled) {
        g_scene.setCompanion(Vec3{}, 0.f, false);
        return;
    }
    ensureCompanionBody();
    if (g_companion.body.IsInvalid())
        return;

    // It stands on the slab; without one it stays where it is instead of
    // falling forever.
    const bool FLOOR = !g_floorBody.IsInvalid();
    if (!FLOOR && g_companion.mode == ECompanionMode::Walk)
        companionHalt("aborted", "There is no floor any more (world.grid is off)");
    g_bodyIf->SetGravityFactor(g_companion.body, FLOOR ? 1.f : 0.f);

    const auto P = g_bodyIf->GetPosition(g_companion.body);
    const Vec3 AT{P.GetX(), P.GetY(), P.GetZ()};
    const auto V = g_bodyIf->GetLinearVelocity(g_companion.body);
    float vx = 0.f, vz = 0.f;

    if (g_companion.mode != ECompanionMode::Idle) {
        Vec3 point{};
        std::optional<Vec3> face;
        float near = kNearPoint;
        std::string why;

        if (!companionGoal(g_companion.kind, g_companion.windowId, g_companion.target,
                           AT, point, face, near, why) ||
            (g_companion.mode == ECompanionMode::Walk &&
             !companionReachable(g_companion.kind, g_companion.target, point, why)))
            companionHalt("aborted", why);

        if (g_companion.mode == ECompanionMode::Turn) {
            if (companionTurn(AT, face.value_or(point), dt))
                g_companion.mode = ECompanionMode::Idle;
        } else if (g_companion.mode == ECompanionMode::Walk) {
            const float SX = AT.x - g_companion.start.x, SZ = AT.z - g_companion.start.z;
            if (!g_companion.stepped && SX * SX + SZ * SZ > 1e-4f) {
                g_companion.stepped = true;
                companionEvent("first_step");
            }

            const float TX = point.x - AT.x, TZ = point.z - AT.z;
            const float DIST = std::sqrt(TX * TX + TZ * TZ);

            if (DIST <= near) {
                // There: it stands, then turns to the target.
                g_companion.lastSpeed = 0.f;
                if (!face || companionTurn(AT, *face, dt)) {
                    companionEvent("arrived");
                    g_companion.mode = ECompanionMode::Idle;
                }
            } else {
                const float DX = TX / DIST, DZ = TZ / DIST;
                float speed = std::clamp(2.5f * (DIST - near) + 0.5f, 0.5f, kCompanionSpeed);
                std::string blocker;

                // Never into the player, never off the edge: it stops short.
                if (g_companion.kind != ECompanionTarget::Player &&
                    !g_playerBody.IsInvalid()) {
                    const auto PP = g_bodyIf->GetPosition(g_playerBody);
                    const float QX = PP.GetX() - AT.x, QZ = PP.GetZ() - AT.z;
                    if (QX * QX + QZ * QZ < kPersonalSpace * kPersonalSpace &&
                        QX * DX + QZ * DZ > 0.f &&
                        std::fabs(PP.GetY() - AT.y) < Camera::kBodyHeight) {
                        speed   = 0.f;
                        blocker = "The player is in the way";
                    }
                }
                const float EDGE = kFloorHalf - Camera::kBodyHalfWidth;
                if (std::fabs(AT.x + DX * 0.35f) > EDGE ||
                    std::fabs(AT.z + DZ * 0.35f) > EDGE) {
                    speed   = 0.f;
                    blocker = "The floor ends here";
                }

                // Progress on the velocity the solver left: what a wall or a
                // prop takes away is missing there.
                const float MADE = V.GetX() * DX + V.GetZ() * DZ;
                if (speed == 0.f || MADE < 0.25f * g_companion.lastSpeed)
                    g_companion.stuck += dt;
                else
                    g_companion.stuck = 0.f;
                g_companion.lastSpeed = speed;

                if (g_companion.stuck > kStuckSeconds)
                    companionHalt("blocked",
                                  blocker.empty() ? "Something is in the way" : blocker);
                else {
                    vx = DX * speed;
                    vz = DZ * speed;
                    companionTurn(AT, point, dt);
                }
            }
        }
    }

    g_bodyIf->SetLinearVelocity(
        g_companion.body,
        JPH::Vec3(vx, FLOOR ? std::max(V.GetY(), -40.f) : 0.f, vz));
    g_scene.setCompanion(AT, g_companion.yaw, true);

    // The eye rides where the player's does, level, along its facing.
    if (g_sightAsked) {
        Camera eye;
        eye.position = Vec3{AT.x, AT.y + PLAYER_EYE_OFF, AT.z};
        eye.yaw      = g_companion.yaw;
        eye.pitch    = 0.f;
        g_scene.requestSight(eye);
        g_sightAsked   = false;
        g_sightTakenNs = monotonicNs();
    }
}

static void sightEvent(const std::string& fields) {
    postRoomEvent("{\"event\":\"sight\"," + fields + ",\"t\":" +
                  std::to_string(monotonicNs()) + "}");
}

// Forgets a picture asked for, and says why -- a bridge waiting for it
// must not wait forever.
static void abortSight(const std::string& why) {
    if (!g_sightPending)
        return;
    g_sightAsked   = false;
    g_sightPending = false;
    g_scene.dropSight();
    sightEvent("\"error\":" + jsonString(why));
}

// Once per frame: the eye's picture as a JPEG next to Hyprland's sockets,
// and the event that says where. Written whole and renamed into place, so a
// reader never sees half of it. The encoding takes 0.6 ms for a picture of
// the station, 1.1 ms for noise (measured 2026-10-07, libjpeg-turbo 3) --
// once a second at most, so it runs here rather than in a thread of its own.
static void deliverSight() {
    if (!g_sightPending)
        return;
    if (!roomOpen()) {
        abortSight("The room closed before the picture was taken");
        return;
    }

    std::vector<unsigned char> rgba;
    std::string why;
    switch (g_scene.takeSight(rgba, why)) {
        case GLScene::ESight::None:
            if (monotonicNs() - g_sightAskedNs > 2'000'000'000LL)
                abortSight("No picture came back within 2 s");
            return;
        case GLScene::ESight::Failed:
            abortSight(why);
            return;
        case GLScene::ESight::Picture: break;
    }

    constexpr int W = GLScene::kSightWidth, H = GLScene::kSightHeight;
    const std::unique_ptr<void, decltype(&tj3Destroy)> TJ{tj3Init(TJINIT_COMPRESS), tj3Destroy};
    unsigned char* jpeg = nullptr;
    size_t size = 0;
    if (!TJ || tj3Set(TJ.get(), TJPARAM_QUALITY, 80) != 0 ||
        tj3Set(TJ.get(), TJPARAM_SUBSAMP, TJSAMP_420) != 0 ||
        tj3Set(TJ.get(), TJPARAM_BOTTOMUP, 1) != 0 ||
        tj3Compress8(TJ.get(), rgba.data(), W, 0, H, TJPF_RGBA, &jpeg, &size) != 0) {
        tj3Free(jpeg);
        abortSight(std::string("JPEG: ") + (TJ ? tj3GetErrorStr(TJ.get()) : "tj3Init failed"));
        return;
    }

    // Hyprland's instance directory, where its sockets are.
    const char* RUNTIME = std::getenv("XDG_RUNTIME_DIR");
    const char* SIG     = std::getenv("HYPRLAND_INSTANCE_SIGNATURE");
    if (!RUNTIME || !SIG) {
        tj3Free(jpeg);
        abortSight("XDG_RUNTIME_DIR or HYPRLAND_INSTANCE_SIGNATURE is not set");
        return;
    }
    const std::string PATH = std::string(RUNTIME) + "/hypr/" + SIG + "/hypr3d-sight.jpg";
    const std::string TMP  = PATH + ".part";
    std::FILE* f = std::fopen(TMP.c_str(), "wb");
    const bool WRITTEN = f && std::fwrite(jpeg, 1, size, f) == size;
    const bool CLOSED  = f && std::fclose(f) == 0;
    tj3Free(jpeg);
    if (!WRITTEN || !CLOSED || std::rename(TMP.c_str(), PATH.c_str()) != 0) {
        std::remove(TMP.c_str());
        abortSight("Cannot write " + PATH + ": " + std::strerror(errno));
        return;
    }

    g_sightPending = false;
    sightEvent("\"path\":" + jsonString(PATH) + ",\"width\":" + std::to_string(W) +
               ",\"height\":" + std::to_string(H) + ",\"fov\":90,\"bytes\":" +
               std::to_string(size) + ",\"taken\":" + std::to_string(g_sightTakenNs));
}

// The room event: open or closed, and when open what can be walked to.
static void postRoomState() {
    const bool OPEN = roomOpen();
    std::string json = std::string("{\"event\":\"room\",\"open\":") + (OPEN ? "true" : "false");
    if (OPEN) {
        json += ",\"targets\":[{\"name\":\"player\"},{\"name\":\"spawn\"}";
        for (const auto& [ID, NAME] : companionWindows()) {
            json += ",{\"name\":" + jsonString(NAME);
            if (const auto W = Compat::findWindowById(ID))
                json += ",\"title\":" + jsonString(W->m_title);
            json += "}";
        }
        json += "]";
    }
    postRoomEvent(json + ",\"t\":" + std::to_string(monotonicNs()) + "}");
}

// Posts the room event when it opened, closed or a window came or went --
// once the change has held for 100 ms, so the frames in which the room fills
// up after opening send one event, not one per window. Titles change all the
// time and do not count.
static std::string                           g_roomPosted = "closed";
static std::string                           g_roomPending;
static std::chrono::steady_clock::time_point g_roomPendingSince{};

static void syncRoomState(bool now) {
    const bool OPEN = roomOpen();
    if (!OPEN && g_companion.mode != ECompanionMode::Idle)
        companionHalt("aborted", "The room was closed");

    std::string sig = OPEN ? "open" : "closed";
    if (OPEN)
        for (const auto& [ID, NAME] : companionWindows())
            sig += "\n" + NAME;

    if (sig == g_roomPosted) {
        g_roomPending.clear();
        return;
    }
    const auto NOW = std::chrono::steady_clock::now();
    if (!now) {
        if (sig != g_roomPending) {
            g_roomPending      = sig;
            g_roomPendingSince = NOW;
            return;
        }
        if (NOW - g_roomPendingSince < std::chrono::milliseconds(100))
            return;
    }
    g_roomPosted = sig;
    g_roomPending.clear();
    postRoomState();
}

// Euler (degrees, our RY*RX*RZ convention) -> Jolt quaternion.
static JPH::Quat eulerToQuat(const Vec3& rotationDeg) {
    constexpr float DEG = 3.14159265358979f / 180.0f;
    const float Y = rotationDeg.y * DEG, X = rotationDeg.x * DEG,
                Z = rotationDeg.z * DEG;

    JPH::Quat Q = JPH::Quat::sRotation(JPH::Vec3::sAxisY(), Y);
    Q = Q * JPH::Quat::sRotation(JPH::Vec3::sAxisX(), X);
    Q = Q * JPH::Quat::sRotation(JPH::Vec3::sAxisZ(), Z);
    return Q;
}

// Jolt quaternion -> Euler (degrees), the inverse of eulerToQuat: for
// R = RY(yaw)*RX(pitch)*RZ(roll), pitch = asin(-R12), yaw = atan2(R02, R00),
// roll = atan2(R10, R11) with R in math row-major.
static Vec3 quatToEuler(const JPH::Quat& q) {
    const JPH::Mat44 M = JPH::Mat44::sRotation(q);
    const JPH::Vec3 C0 = M.GetColumn3(0);
    const JPH::Vec3 C1 = M.GetColumn3(1);
    const JPH::Vec3 C2 = M.GetColumn3(2);

    constexpr float RAD = 180.0f / 3.14159265358979f;

    return Vec3{
        std::asin(std::clamp(-C2.GetY(), -1.0f, 1.0f)) * RAD,
        std::atan2(C2.GetX(), C2.GetZ()) * RAD,
        std::atan2(C0.GetY(), C1.GetY()) * RAD,
    };
}

static void joltSyncBodies() {
    g_joltBodies.resize(g_sceneObjects.size());
    g_shapeJobs.resize(g_sceneObjects.size());

    for (size_t i = 0; i < g_sceneObjects.size(); ++i) {
        auto& JB = g_joltBodies[i];
        const auto& OBJ = g_sceneObjects[i];
        const auto* MODEL = g_scene.sceneModel(i);

        const bool WANT = OBJ.collision && MODEL && MODEL->loaded() &&
            !MODEL->triangles().empty();
        const uint32_t MESH_VER = MODEL ? MODEL->meshVersion() : 0;

        if (!WANT) {
            g_shapeJobs[i].reset();
            if (JB.valid) {
                g_bodyIf->RemoveBody(JB.body);
                g_bodyIf->DestroyBody(JB.body);
                JB = {};
            }
            continue;
        }

        // Rebuild ONLY on a mesh-content change (or a body-space mapping
        // change: scale/pivot edits): the generation also bumps on every
        // transform move, and a falling object would otherwise recreate its
        // ConvexHull every frame (the 0.5s-per-tick freeze).
        const Vec3 SCL = MODEL->scale();
        const Vec3 PIV = MODEL->pivot();
        const bool MAPPING_CHANGED =
            JB.builtScale.x != SCL.x || JB.builtScale.y != SCL.y ||
            JB.builtScale.z != SCL.z || JB.builtPivot.x != PIV.x ||
            JB.builtPivot.y != PIV.y || JB.builtPivot.z != PIV.z;

        if (!JB.valid || JB.meshGen != MESH_VER || MAPPING_CHANGED) {
            // Body-space triangles: the mesh scaled, with the PIVOT at the
            // body origin. The body transform (config position/rotation)
            // then maps body space onto the world exactly like the render
            // matrix does -- collision follows the visual for ANY config
            // transform, and a center_offset edit only moves the pivot.
            // (World-baked vertices here would apply the body transform a
            // SECOND time: an object placed at {2,0,1} got its collision at
            // {4,0,2}.)
            const auto BODY_PT = [SCL, PIV](const Vec3& p) {
                return Vec3{SCL.x * p.x - SCL.x * PIV.x,
                            SCL.y * p.y - SCL.y * PIV.y,
                            SCL.z * p.z - SCL.z * PIV.z};
            };

            const auto& LOCAL = MODEL->localTriangles();

            const bool DYN = OBJ.dynamic && OBJ.physics;
            const JPH::EMotionType MOTION =
                OBJ.dynamic ? (OBJ.physics ? JPH::EMotionType::Dynamic
                                           : JPH::EMotionType::Kinematic)
                            : JPH::EMotionType::Static;
            const JPH::ObjectLayer LAYER =
                MOTION == JPH::EMotionType::Static ? LAYER_STATIC : LAYER_MOVING;

            JPH::Ref<JPH::Shape> SHAPE;
            if (MOTION == JPH::EMotionType::Static) {
                // Static meshes keep their exact triangle soup, cooked on a
                // worker. The old body stays until the new shape is there.
                auto& JOB = g_shapeJobs[i];
                const bool CURRENT = JOB && JOB->meshGen == MESH_VER &&
                    JOB->scale.x == SCL.x && JOB->scale.y == SCL.y &&
                    JOB->scale.z == SCL.z && JOB->pivot.x == PIV.x &&
                    JOB->pivot.y == PIV.y && JOB->pivot.z == PIV.z;

                if (!CURRENT) {
                    JOB.reset(); // an outdated one is waited for
                    JOB          = std::make_unique<SShapeJob>();
                    JOB->meshGen = MESH_VER;
                    JOB->scale   = SCL;
                    JOB->pivot   = PIV;
                    JOB->thread  = std::jthread([J = JOB.get(), LOCAL, BODY_PT] {
                        // An exception leaving this thread would terminate
                        // Hyprland.
                        try {
                            JPH::TriangleList TL;
                            // Two-sided: each triangle twice (both windings)
                            // -- Jolt's narrow phase ignores back faces, and
                            // single-sided authored geometry (ceilings!)
                            // would let bodies tunnel through.
                            TL.reserve(LOCAL.size() * 2);
                            for (const auto& T : LOCAL) {
                                const Vec3 A = BODY_PT(T.a), B = BODY_PT(T.b),
                                           C = BODY_PT(T.c);
                                TL.push_back(JPH::Triangle(
                                    JPH::Float3(A.x, A.y, A.z),
                                    JPH::Float3(B.x, B.y, B.z),
                                    JPH::Float3(C.x, C.y, C.z)));
                                TL.push_back(JPH::Triangle(
                                    JPH::Float3(A.x, A.y, A.z),
                                    JPH::Float3(C.x, C.y, C.z),
                                    JPH::Float3(B.x, B.y, B.z)));
                            }

                            JPH::MeshShapeSettings SETTINGS(TL);
                            SETTINGS.SetEmbedded();
                            auto RES = SETTINGS.Create();
                            if (!RES.HasError())
                                J->shape = RES.Get();
                        } catch (...) {
                            J->shape = nullptr;
                        }
                        J->done.store(true, std::memory_order_release);
                    });
                    continue;
                }

                // Still cooking, or Jolt refused this mesh (kept, so it is
                // not tried again until the mesh changes).
                if (!JOB->done.load(std::memory_order_acquire) || !JOB->shape)
                    continue;

                JOB->thread.join();
                SHAPE = JOB->shape;
                JOB.reset();
            } else {
                if (JB.valid) {
                    g_bodyIf->RemoveBody(JB.body);
                    g_bodyIf->DestroyBody(JB.body);
                    JB = {};
                }

                // Dynamic/kinematic bodies use a convex hull (Jolt requires
                // it; the hull also tumbles believably).
                JPH::Array<JPH::Vec3> POINTS;
                POINTS.reserve(LOCAL.size() * 3);
                for (const auto& T : LOCAL)
                    for (const Vec3* P : {&T.a, &T.b, &T.c}) {
                        const Vec3 Q = BODY_PT(*P);
                        POINTS.push_back(JPH::Vec3(Q.x, Q.y, Q.z));
                    }

                JPH::ConvexHullShapeSettings HS(POINTS);
                HS.SetEmbedded();
                auto RES = HS.Create();
                if (RES.HasError())
                    continue;
                SHAPE = RES.Get();
            }

            if (JB.valid) {
                g_bodyIf->RemoveBody(JB.body);
                g_bodyIf->DestroyBody(JB.body);
                JB = {};
            }

            JPH::BodyCreationSettings BCS(
                SHAPE,
                JPH::RVec3(OBJ.position.x, OBJ.position.y, OBJ.position.z),
                eulerToQuat(OBJ.rotationDeg), MOTION,
                i == g_mapGrabIndex ? LAYER_GRABBED : LAYER);
            BCS.mLinearDamping  = 0.05f;
            BCS.mAngularDamping = 0.05f;
            BCS.mAllowSleeping  = true;
            BCS.mFriction       = 0.6f;
            BCS.mRestitution    = 0.05f;

            JPH::Body* B = g_bodyIf->CreateBody(BCS);
            g_bodyIf->AddBody(B->GetID(), JPH::EActivation::Activate);

            JB.body       = B->GetID();
            JB.meshGen    = MESH_VER;
            JB.builtScale = SCL;
            JB.builtPivot = PIV;
            JB.valid      = true;
        } else {
            // Config-owned objects (static/kinematic/no-physics): follow the
            // config transform if it moved (grab, config edit).
            const auto WANT = JPH::RVec3(OBJ.position.x, OBJ.position.y,
                                         OBJ.position.z);
            const auto CUR = g_bodyIf->GetPosition(JB.body);

            g_bodyIf->SetPositionAndRotationWhenChanged(
                JB.body, WANT, eulerToQuat(OBJ.rotationDeg),
                JPH::EActivation::DontActivate);
        }
    }
}

// Nearest scene-model hit of a ray. Occlusion (windows behind models lose
// clicks/focus/aim) and the model gestures both go through this: collision-
// enabled objects answer through their BVH trees (fast), a carried object is
// skipped (it is on the crosshair already). collision = false objects neither
// occlude nor grab.
struct SModelRayHit {
    bool   hit = false;
    size_t index = SIZE_MAX;
    float  dist = 0.0f;
    bool   dynamic = false;
};

static SModelRayHit modelRayHit(const Vec3& origin, const Vec3& dir,
                                bool dynamicOnly) {
    SModelRayHit R;

    for (size_t i = 0; i < g_sceneObjects.size(); ++i) {
        if (i >= g_objTrees.size())
            break; // config reloaded mid-frame: the trees lag one resize

        if (dynamicOnly && !g_sceneObjects[i].dynamic)
            continue;
        if (!g_sceneObjects[i].collision)
            continue;
        if (i == g_mapGrabIndex)
            continue;

        float t = -1.f;
        if (!g_objTrees[i].empty())
            t = g_objTrees[i].rayCast(origin, dir);
        else if (const auto* MODEL = g_scene.sceneModel(i);
                 MODEL && MODEL->loaded())
            t = MODEL->rayCast(origin, dir);

        if (t > 0.f && (!R.hit || t < R.dist)) {
            R.hit     = true;
            R.index   = i;
            R.dist    = t;
            R.dynamic = g_sceneObjects[i].dynamic;
        }
    }

    return R;
}

// A soft shadow on the floor under every window (GLScene::setShadows), the
// grounding the Reddit finds asked for: a window floating over nothing reads
// as nowhere. The floor is where a ray straight down meets the room -- the
// map, or the grid's slab (top at 0, -20..20). The blob covers the window's
// footprint; the higher the window floats, the fainter and wider it gets, as
// a shadow's penumbra grows with the gap. It is flat, so it is drawn only
// over flat floor: under each corner of the footprint the floor lies within
// kShadowFlat of the floor under the middle. A blob half over the Moon
// station's deck edge hung in the air and darkened the edge's light across
// the whole view (measured 2026-10-07); over an edge, a step or rough ground
// there is no shadow rather than a wrong one.
static constexpr float kShadowAlpha  = 0.55f; // a window standing on the floor
static constexpr float kShadowFade   = 1.0f;  // m of height that take it to 1/e
static constexpr float kShadowSpread = 0.15f; // m of blur at the floor, +0.25 per m up
static constexpr float kShadowFlat   = 0.15f; // m the floor may vary under it

// The floor's height under (x, z), looking down from y; false: none.
static bool floorBelow(float x, float y, float z, float& floorY) {
    const auto DOWN = modelRayHit(Vec3{x, y, z}, Vec3{0.f, -1.f, 0.f}, /*dynamicOnly=*/false);
    if (DOWN.hit) {
        floorY = y - DOWN.dist;
        return true;
    }
    if (g_cfgGrid && y > 0.0f && std::fabs(x) < 20.0f && std::fabs(z) < 20.0f) {
        floorY = 0.0f;
        return true;
    }
    return false;
}

static void updateShadows() {
    std::vector<GLScene::SShadow> out;
    for (const auto& E : g_world.entities()) {
        if (!g_cfgShadows)
            break;
        // Windows only: the rice's layers (its wallpaper, its bar) are in the
        // world too, as wide as the wall, and cast a band across the view.
        if (E.width <= 0.0f || E.height <= 0.0f || !Compat::findWindowById(E.id))
            continue;
        float floorY = 0.0f;
        if (!floorBelow(E.center.x, E.center.y, E.center.z, floorY))
            continue;
        const Vec3  R = g_world.rightOf(E.id), U = g_world.upOf(E.id);
        const float HR = std::hypot(R.x, R.z), HU = std::hypot(U.x, U.z);
        const float BOTTOM = E.center.y - std::fabs(U.y) * E.height * 0.5f - std::fabs(R.y) * E.width * 0.5f;
        const float GAP = std::max(0.0f, BOTTOM - floorY);
        const float ALPHA = kShadowAlpha * std::exp(-GAP / kShadowFade);
        if (ALPHA < 0.02f)
            continue;
        const float SPREAD = kShadowSpread + 0.25f * GAP;
        GLScene::SShadow S;
        S.center = Vec3{E.center.x, floorY + 0.01f, E.center.z};
        S.right  = HR > 1e-3f ? Vec3{R.x / HR, 0.f, R.z / HR} : Vec3{1.f, 0.f, 0.f};
        const Vec3 ALONG = S.right * (E.width * 0.5f * HR);
        const Vec3 ACROSS = Vec3{-S.right.z, 0.f, S.right.x} * (E.height * 0.5f * HU + kShadowSpread);
        bool flat = true;
        for (const Vec3 C : {S.center - ALONG - ACROSS, S.center + ALONG - ACROSS,
                             S.center + ALONG + ACROSS, S.center - ALONG + ACROSS}) {
            float y = 0.0f;
            if (!floorBelow(C.x, E.center.y, C.z, y) || std::fabs(y - floorY) > kShadowFlat) {
                flat = false;
                break;
            }
        }
        if (!flat)
            continue;
        // The blob is dark over about its inner half: 1.4x the footprint.
        S.halfW  = E.width * 0.5f * HR * 1.4f + SPREAD;
        S.halfD  = E.height * 0.5f * HU * 1.4f + SPREAD;
        S.alpha  = ALPHA;
        out.push_back(S);
    }
    g_scene.setShadows(std::move(out));
}

// A model is IN FRONT of the aimed window when its ray hit is closer (or
// there is no window hit at all).
static bool modelInFront(const Vec3& origin, const Vec3& dir,
                         const World3D::SHit& windowHit) {
    const auto MR = modelRayHit(origin, dir, /*dynamicOnly=*/false);
    return MR.hit && (!windowHit.hit || MR.dist < windowHit.distance);
}


static void notify(const std::string& text, const CHyprColor& color);

// Feeds the requested path to the scene every frame. Tilde is expanded, a
// missing file is reported once per path, and the scene itself reloads the
// texture when the file's mtime changes.
static void updatePanorama() {
    std::string path = g_cfgPanorama;

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
static double   g_diagLastZoomStep = 0.0; // normalized wheel steps (+ = forward)
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

static std::vector<PHLMONITOR> spannedMonitors();

static void damageCurrentMonitor() {
    if (!g_monitor)
        return;

    if (!g_pHyprRenderer)
        return;

    g_pHyprRenderer->damageMonitor(
        g_monitor
    );

    // The neighbours the room spans show the same moving room.
    if (g_active)
        for (const auto& mon : spannedMonitors())
            if (mon != g_monitor)
                g_pHyprRenderer->damageMonitor(mon);
}

// Every monitor redrawn: leaving the room, a neighbour with no window to
// re-lay out would otherwise keep showing its last 3D frame.
static void damageAllMonitors() {
    if (!g_pHyprRenderer || !State::monitorState())
        return;

    for (const auto& mon : State::monitorState()->monitors())
        if (mon)
            g_pHyprRenderer->damageMonitor(mon);
}

static PHLMONITOR g_currentRenderMon = nullptr;

// The monitor the room was entered from: the eye stays in front of it while
// the room is open. The crosshair gives the aimed window the focus, and
// Hyprland moves the monitor focus to that window's monitor -- following it,
// the view was rebuilt around the top or the side monitor whenever a window
// laid out there came under the crosshair, and the crosshair went with it
// (measured on the owner's three monitors).
static PHLMONITORREF g_viewMonitor;

// The monitor the 3D view is built for: configured monitor, the one the room
// was entered from, or the focused monitor, falling back to whatever
// render.pre last reported.
static PHLMONITOR targetMonitor() {
    if (!g_cfgMonitor.empty() && State::monitorState()) {
        for (const auto& mon : State::monitorState()->monitors()) {
            if (mon && mon->m_name == g_cfgMonitor)
                return mon;
        }
    }

    if (const auto VIEW = g_viewMonitor.lock(); g_active && VIEW && VIEW->m_enabled)
        return VIEW;

    if (const auto FOCUSED = Compat::focusedMonitor())
        return FOCUSED;

    return g_monitor;
}

// The monitors the room spans: with world.span (the default) every enabled
// monitor that mirrors none, the one the view is built for first; without it
// that one alone.
static std::vector<PHLMONITOR> spannedMonitors() {
    std::vector<PHLMONITOR> out;
    const auto MAIN = targetMonitor();

    if (!MAIN)
        return out;

    out.push_back(MAIN);

    if (!g_cfgSpan || !State::monitorState())
        return out;

    for (const auto& mon : State::monitorState()->monitors()) {
        if (!mon || mon == MAIN || !mon->m_enabled || mon->m_isUnsafeFallback ||
            mon->m_mirrorOf.lock() || mon->m_size.x <= 0 || mon->m_size.y <= 0)
            continue;

        out.push_back(mon);
    }

    return out;
}

static bool isSpanned(const PHLMONITOR& mon) {
    const auto ALL = spannedMonitors();
    return std::find(ALL.begin(), ALL.end(), mon) != ALL.end();
}

// The view plane the monitors are windows in. The eye sits in front of the
// centre of the monitor the view is built for, at the distance (in logical
// px) at which that monitor spans the field of view; every monitor lies in the
// one plane where the layout puts it. So the room continues across the
// screens the way the desktop does -- the layout's arrangement, flat: tilted
// side screens and bezels are not modelled. (Kooima, "Generalized
// Perspective Projection", for a flat wall of screens.)
struct SViewPlane {
    double distance = 0.0;      // eye to plane, logical px
    double cx = 0.0, cy = 0.0;  // the eye's foot on the plane, layout px
};

static std::optional<SViewPlane> viewPlane() {
    const auto MAIN = targetMonitor();

    if (!MAIN || MAIN->m_size.x <= 0 || MAIN->m_size.y <= 0)
        return std::nullopt;

    const double TANFOV = std::tan(kFovDeg * std::numbers::pi / 360.0);

    return SViewPlane{
        MAIN->m_size.y * 0.5 / TANFOV,
        MAIN->m_position.x + MAIN->m_size.x * 0.5,
        MAIN->m_position.y + MAIN->m_size.y * 0.5,
    };
}

// The rectangle of the view plane a monitor shows, in tangents of the view
// axis. For the main monitor it is its own symmetric field of view.
static GLScene::ViewWindow viewWindowFor(const PHLMONITOR& mon) {
    const auto PLANE = viewPlane();

    if (!PLANE || !mon)
        return {};

    const double D = PLANE->distance;

    return GLScene::ViewWindow{
        static_cast<float>((mon->m_position.x - PLANE->cx) / D),
        static_cast<float>((mon->m_position.x + mon->m_size.x - PLANE->cx) / D),
        static_cast<float>((PLANE->cy - mon->m_position.y - mon->m_size.y) / D),
        static_cast<float>((PLANE->cy - mon->m_position.y) / D),
    };
}

// The windows and panels of every monitor the room spans, each with its
// monitor.
static std::vector<Compat::SWindowInfo> eligibleWindowsSpanned() {
    std::vector<Compat::SWindowInfo> out;

    for (const auto& mon : spannedMonitors()) {
        auto part = Compat::enumerateEligibleWindows(mon);
        out.insert(out.end(), std::make_move_iterator(part.begin()),
                   std::make_move_iterator(part.end()));
    }

    return out;
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
    // In2D is the fullscreen passthrough: Hyprland owns everything.
    return g_active && g_transition >= 0.9f &&
        g_fsPhase != EFullscreenPhase::In2D && !sessionLocked();
}

static void setFocusLock(bool on) {
    const CHyprColor COLOR{0.2f, 0.8f, 0.4f, 1.0f};
    if (!on) {
        if (g_focusLockId == 0)
            return;
        // g_lastFocusId stays the held window, which has the keyboard: when
        // the crosshair is on nothing the next frame drops it. Set to 0 here,
        // nothing changed -- measured, the window kept the focus.
        g_focusLockId = 0;
        notify("[hypr3d] focus lock off: the keyboard follows the crosshair again", COLOR);
        return;
    }
    const auto WINDOW = Compat::focusedWindow();
    if (!WINDOW) {
        notify("[hypr3d] focus lock: no window has the keyboard to keep", COLOR);
        return;
    }
    g_focusLockId = Compat::windowId(WINDOW);
    notify("[hypr3d] focus lock on: " + WINDOW->m_title + " keeps the keyboard (F6 releases)", COLOR);
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
// INFOS: the windows of every monitor the room spans (eligibleWindowsSpanned):
// one first pass saves them all, or the second monitor's windows would be
// taken for windows new to the room and shrunk to a spawn panel.
static void ghostWindows(const std::vector<Compat::SWindowInfo>& INFOS) {
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
        // user actually worked with. setWindowBox takes GLOBAL layout
        // coordinates: centre on this monitor, not on the layout origin
        // (which belongs to whichever monitor sits at 0,0, or none at all).
        // A window that floats already is no tile: it keeps the size its
        // rule or its client gave it -- the room's menu came out 960x540 and
        // cut off, a dialog would come out as large.
        if (info.window && info.monitor && !info.floating) {
            const auto& mon = info.monitor;
            const double CX =
                mon->m_position.x + mon->m_size.x * 0.5 - kSpawnWidth * 0.5;
            const double CY =
                mon->m_position.y + mon->m_size.y * 0.5 - kSpawnHeight * 0.5;

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

// Rendering on demand (frameFingerprint, below): frames in a row that showed
// nothing new, and the pump's pace once the room is still.
static constexpr int  kStillFrames      = 3;
static constexpr auto kIdlePumpInterval = std::chrono::milliseconds(50);
static uint64_t       g_frameSig        = 0;
static int            g_stillFrames     = 0;
static bool           runsOnTime();
static bool           roomStill();
static void           dumpStatus(bool force);

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

                // 2D passthrough: no damage (Hyprland renders by its own
                // damage); the tick still polls the fullscreen state so the
                // exit transition can fire.
                pollFullscreen();

                // Snapshot release deferred from deactivate3D: safe to
                // destroy GL objects here, outside any render pass.
                if (g_captureReleasePending && !g_active) {
                    g_capture.releaseAll();
                    g_captureReleasePending = false;
                }

                if (g_fsPhase != EFullscreenPhase::In2D && !roomStill())
                    damageCurrentMonitor();
                else if (roomStill())
                    dumpStatus(false); // no frame writes it: a still room's last state, still read

                self->updateTimeout(roomStill() ? kIdlePumpInterval : kFramePumpInterval);
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

    bool consumedSkirt = false;

    for (const auto& info : infos) {
        // Focus-change feedback (the active/inactive opacity fade and the
        // border color tween) is compositor-side -- no client commit happens,
        // so the buffer-change check would freeze both animations mid-way
        // and make the window snap to its final look. While any of them
        // runs, the snapshot refreshes at full rate. The FADE and FULLSCREEN
        // channels belong to the same family: the fullscreen handoff fades
        // every non-FS window through them, and without this the last
        // snapshot keeps the MID-FADE alpha baked in -- static windows
        // stayed semi-transparent until a hover forced a repaint.
        // Alpha channels in flight force full-rate retakes -- but the LAST
        // retake must happen AFTER the animation finishes: while
        // isBeingAnimated() is still true the value can be 0.99, the
        // animation then completes and no further retake runs, leaving the
        // snapshot baked one tick short of the goal (the lingering
        // semi-transparency on static windows). So once a channel was seen
        // in flight, the window keeps forcing retakes for a grace period
        // past the end -- the final retake captures the goal state exactly.
        static std::unordered_map<std::uintptr_t,
            std::chrono::steady_clock::time_point> alphaGrace;

        const auto NOW = std::chrono::steady_clock::now();

        const bool FADING = info.window &&
            (info.window->alpha(Desktop::View::WINDOW_ALPHA_ACTIVE)
                 ->isBeingAnimated() ||
                info.window->alpha(Desktop::View::WINDOW_ALPHA_FADE)
                 ->isBeingAnimated() ||
                info.window->alpha(Desktop::View::WINDOW_ALPHA_FULLSCREEN)
                 ->isBeingAnimated() ||
                info.window->m_borderFadeAnimationProgress->isBeingAnimated());

        if (FADING)
            alphaGrace[info.id] = NOW + std::chrono::milliseconds(200);
        else if (auto IT = alphaGrace.find(info.id);
                 IT != alphaGrace.end() && NOW > IT->second)
            alphaGrace.erase(IT); // expired: keep the map from growing

        const bool ALPHA_GRACE = info.window &&
            alphaGrace.count(info.id) > 0;

        const bool FORCE =
            FADING ||
            ALPHA_GRACE ||
            info.id == g_lastAimedId ||
            (g_world.dragActive() && g_world.draggedId() == info.id) ||
            (g_resize.active && g_resize.id == info.id) ||
            info.id == FOCUSED_ID;

        bool consumedSkirt = false;

        // The frame after a resize gesture ended: one forced snapshot, whose
        // only purpose is the settled full-resolution silhouette refresh.
        bool finalSkirt = false;
        if (info.id == g_skirtFinalRefreshId) {
            finalSkirt = true;
            consumedSkirt = true;
        }

        if (info.isLayer)
            g_capture.makeSnapshotLayer(info.layer, info.monitor ? info.monitor : mon, FORCE || finalSkirt);
        else
            g_capture.makeSnapshot(info.window, info.monitor ? info.monitor : mon, FORCE || finalSkirt);
    }

    if (consumedSkirt)
        g_skirtFinalRefreshId = 0;

    ++g_captureFrames;
}

// Called from render.pre. Hyprland emits render.pre before beginRender() and
// before the compositor's main render pass, so makeSnapshotFB can safely open
// its own offscreen render here. Nested snapshot renders set g_capturing and
// therefore skip all plugin rendering callbacks.
static void serviceCapture() {
    if (!g_active || g_capturing || !g_pHyprRenderer)
        return;

    // 2D passthrough: the room is dormant, no captures needed.
    if (g_fsPhase == EFullscreenPhase::In2D)
        return;

    const auto MON = targetMonitor();
    if (!MON)
        return;

    g_capturing = true;

    ghostWindows(eligibleWindowsSpanned());
    refreshCaptures(eligibleWindowsSpanned(), MON);

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

    // A scene model in front of the aimed window occludes it: no window
    // focus through geometry (one distance space for both classes).
    std::uintptr_t AIMED = HIT.hit ? HIT.id : 0;
    if (AIMED != 0 && modelInFront(cam.position, cam.centerRay(), HIT))
        AIMED = 0;
    g_lastAimedId = AIMED;

    const std::uintptr_t FOCUS = g_aim.update(AIMED, dt);

    // The lock outranks the crosshair; Hyprland's border on the held window
    // is its visible frame, as no other window gets the focus.
    if (g_focusLockId != 0) {
        const auto LOCKED = Compat::findWindowById(g_focusLockId);
        if (LOCKED) {
            if (Compat::focusedWindow() != LOCKED)
                Compat::focusWindow(LOCKED);
            g_lastFocusId = g_focusLockId;
            return;
        }
        // Its window is gone, and Hyprland has given the keyboard to some
        // other window on the close -- measured, while the crosshair was on
        // nothing. From the gone window the lines below go back to what the
        // crosshair says, as they do when an unlocked window closes.
        g_lastFocusId = g_focusLockId;
        g_focusLockId = 0;
        notify("[hypr3d] focus lock off: its window is gone", CHyprColor{0.2f, 0.8f, 0.4f, 1.0f});
    }

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

// Raw (pre-smoothstep) fullscreen transition progress. Shared so syncWorld
// and applyFullscreenAnimation agree on the same instant.
static float fsRawProgress() {
    return std::clamp(
        std::chrono::duration<float>(std::chrono::steady_clock::now() -
                                     g_fsPhaseStart).count() /
            kFsAnimDuration,
        0.0f, 1.0f);
}

// The room's uniform window scale. Shared by syncWorld (quad size) and the
// fullscreen animation (its endpoints must match what the room renders, or
// the transition visibly jumps).
// The wall's scale (world.span): the desktop enters 1:1 at any distance as
// long as its size shrinks with it -- the same angle, the same pixels. At
// world.window_scale 0.5 a 1080 px monitor stands 4.2 m off, and on the
// owner's three monitors its lower third was under the Moon station's deck
// (measured 2026-10-07). So as the wall is built its scale is capped where
// its lowest monitor ends just above the player's feet; it stays until
// reset(), so the windows already on it keep their size. 0: not built yet.
static float           g_wallFit      = 0.0f;
static constexpr float kWallFloorGap  = 0.05f; // m between the wall and the feet

static float configWindowScale() {
    return g_wallFit > 0.0f ? std::min(g_cfgWindowScale, g_wallFit) : g_cfgWindowScale;
}

// F2: the aimed window comes to the eye at 1:1 -- one of its pixels on one
// pixel of the screen, its texels on the pixel centres -- and F2 again sends
// it back where it stood. The window moves, not the camera, the way a
// fullscreen window comes to the screen: the player stays where he stands
// and nothing collides. On the way there the reading pose follows the live
// camera; once there it stays put in the room -- the menu follows the player
// (followMenu). F4 is the cinema: the same
// way, but the window as large as the view takes it and the room dark
// around it, as a cinema dims its lights.
struct SRead {
    std::uintptr_t id      = 0;
    bool           cinema  = false; // F4, not F2
    bool           back    = false; // on the way back to the room
    bool           arrived = false;
    float          legDim  = 0.f, dim = 0.f; // the room's dark, 0..1
    std::chrono::steady_clock::time_point start;
    Vec3  legCenter{};      // the pose the current leg starts from
    float legYaw = 0.f, legPitch = 0.f, legRoll = 0.f, legScale = 1.f;
    Vec3  roomCenter{};     // where it stood in the room
    float roomYaw = 0.f, roomPitch = 0.f, roomRoll = 0.f;
    Vec3  atCenter{};       // the reading pose, fixed on arrival
    float atYaw = 0.f, atPitch = 0.f, atScale = 1.f;
    bool  following = false; // the menu, gliding back in front (followMenu)
    std::chrono::steady_clock::time_point tick; // followMenu's last frame
};
static SRead g_read;
// The HUD (config world.hud): the rice's top and overlay layers -- its bar,
// its notifications, its drawers -- over the picture where they stand in 2D,
// in view wherever the player looks, as a game's HUD is. On the wall a
// notification came where the 2D picture had it, out of sight behind the
// player; the Reddit finds report the same of Xreal's virtual screens.
// caelestia draws bar and notifications into ONE screen-sized layer, so they
// come over the picture together; background and bottom layers (the
// wallpaper) stay on the wall. Shown, not aimed at: the crosshair passes
// through to the room, as through the screen-sized layer it would otherwise
// always hit.
struct SHudItem {
    std::uintptr_t id = 0;
    PHLMONITORREF  monitor;
    CBox           box; // monitor-local, logical px
};
static std::vector<SHudItem> g_hudItems;

// The room's menu (config menu = { command, title }): F1 runs the command,
// and the window with this title comes to the eye when it opens.
static std::string    g_cfgMenuCommand, g_cfgMenuTitle;
static std::uintptr_t g_menuId = 0; // the menu window last brought
static constexpr float kReadSeconds = 0.35f;
static constexpr float kCinemaFill  = 0.9f; // of the view's width or height
static constexpr float kCinemaDim   = 0.8f; // the room at a fifth of its light

// The reading pose for a window of BOX (logical px, decorations included):
// at the distance where the monitor's logical height fills the view, a
// window at scale 1 shows one logical px per logical px of the screen (the
// fullscreen transition's distance) -- 8.5 m on a 1080 px monitor, and the
// Moon station's deck edge, 7 m off, hid the lower half. So it comes to
// kReadDistance, scaled down by the same ratio: the same angle, the same
// pixels. Shifted by under a pixel so its edges fall on pixel edges: half a
// pixel off, every texel is smeared over two.
static constexpr float kReadDistance = 1.0f;

static void readingPose(const PHLMONITOR& mon, const CBox& BOX, bool cinema, Vec3& center,
                        float& yaw, float& pitch, float& scale) {
    const auto& CAM = g_scene.camera();
    const Vec3  FWD = CAM.forward();
    const Vec3  RIGHT = CAM.right();
    const Vec3  UP = cross(RIGHT, FWD);
    const float ONE_TO_ONE = World3D::toWorld(static_cast<float>(mon->m_size.y)) * 0.5f /
        std::tan(kFovDeg * 3.14159265f / 360.0f);
    scale = kReadDistance / ONE_TO_ONE;

    const double S  = mon->m_scale > 0.0 ? mon->m_scale : 1.0;
    const double PX = scale / (World3D::LOGICAL_PX_PER_UNIT * S); // world per pixel
    const auto   OFF = [](double screen, double window) {
        const double EDGE = (screen - window) * 0.5;
        return EDGE - std::floor(EDGE); // 0 or 0.5 for whole sizes
    };
    double DX = OFF(mon->m_pixelSize.x, BOX.w * S);
    double DY = OFF(mon->m_pixelSize.y, BOX.h * S);
    if (cinema && BOX.w > 0 && BOX.h > 0) {
        // Not 1:1 any more, so no pixel edges to meet.
        scale *= std::min(kCinemaFill * mon->m_size.x / BOX.w, kCinemaFill * mon->m_size.y / BOX.h);
        DX = DY = 0.0;
    }

    center = CAM.position + FWD * kReadDistance - RIGHT * static_cast<float>(DX * PX) +
        UP * static_cast<float>(DY * PX);
    yaw   = -CAM.yaw;
    pitch = CAM.pitch;
}

// The menu follows the player, the way VR toolkits keep a menu at hand
// (Unity XRI's LazyFollow, MRTK's RadialView): it stands still in the room
// while it is being used, and glides back in front once the player has left
// it -- the crosshair off it, or walked off its reading distance. Still while
// aimed at, because the crosshair is the pointer here: a menu that turned
// with every turn could never be pointed at off its centre. Back in front it
// is at 1:1 again. Carrying it off with Super+LMB leaves it in the room.
static constexpr float kMenuMargin = 0.08f; // m the crosshair may stray past its edge
static constexpr float kMenuLeash  = 0.3f;  // m off the reading distance (XRI's max distance)
static constexpr float kMenuSpeed  = 6.0f;  // 1/s, exponential (XRI's movement speed)

static void followMenu(const PHLMONITOR& mon, const CBox& BOX) {
    const auto  NOW = std::chrono::steady_clock::now();
    const float DT  = std::clamp(std::chrono::duration<float>(NOW - g_read.tick).count(), 0.0f, 0.1f);
    g_read.tick = NOW;
    const auto& CAM = g_scene.camera();
    if (!g_read.following) {
        // Where the crosshair's ray meets the menu's plane, in its own axes
        // (last frame's pose, which at rest is this one).
        const Vec3  O = CAM.position, D = CAM.centerRay();
        const Vec3  N = g_world.normalOf(g_read.id);
        const Vec3  OFF = g_read.atCenter - O;
        const float DN = dot(D, N);
        bool        onIt = false;
        if (std::fabs(DN) > 1e-4f) {
            const float T = dot(OFF, N) / DN;
            const Vec3  HIT = O + D * T - g_read.atCenter;
            const float HW = World3D::toWorld(BOX.w) * g_read.atScale * 0.5f + kMenuMargin;
            const float HH = World3D::toWorld(BOX.h) * g_read.atScale * 0.5f + kMenuMargin;
            onIt = T > 0.0f && std::fabs(dot(HIT, g_world.rightOf(g_read.id))) <= HW &&
                std::fabs(dot(HIT, g_world.upOf(g_read.id))) <= HH;
        }
        const float DIST = std::sqrt(dot(OFF, OFF));
        if (onIt && std::fabs(DIST - kReadDistance) <= kMenuLeash)
            return;
        g_read.following = true;
    }

    Vec3  to{};
    float toYaw = 0.f, toPitch = 0.f, toScale = 1.f;
    readingPose(mon, BOX, false, to, toYaw, toPitch, toScale);
    const float K = 1.0f - std::exp(-kMenuSpeed * DT);
    const float DYAW = std::remainder(toYaw - g_read.atYaw, 2.f * std::numbers::pi_v<float>);
    g_read.atCenter = g_read.atCenter + (to - g_read.atCenter) * K;
    g_read.atYaw += DYAW * K;
    g_read.atPitch += (toPitch - g_read.atPitch) * K;
    g_read.atScale = toScale;
    const Vec3 LEFT = to - g_read.atCenter;
    if (dot(LEFT, LEFT) < 1e-6f && std::fabs(DYAW) < 1e-3f && std::fabs(toPitch - g_read.atPitch) < 1e-3f) {
        g_read.atCenter = to, g_read.atYaw = toYaw, g_read.atPitch = toPitch; // on the pixel grid again
        g_read.following = false;
    }
}

// One window of syncWorld's loop: the reading animation owns its pose and
// size. Applied in the loop, not after it: the draw list is built from these
// values, and a later override reaches the screen a frame late.
static void applyReading(const PHLMONITOR& mon, const CBox& BOX, World3D::SEntity& E) {
    const float RAW = std::clamp(
        std::chrono::duration<float>(std::chrono::steady_clock::now() - g_read.start).count() /
            kReadSeconds,
        0.0f, 1.0f);
    const float P = RAW * RAW * (3.0f - 2.0f * RAW); // smoothstep

    Vec3  to{};
    float toYaw = 0.f, toPitch = 0.f, toRoll = 0.f, toScale = 1.f;
    if (g_read.back) {
        to = g_read.roomCenter;
        toYaw = g_read.roomYaw, toPitch = g_read.roomPitch, toRoll = g_read.roomRoll;
        toScale = configWindowScale();
    } else if (g_read.arrived) {
        if (g_read.id == g_menuId && !g_read.cinema)
            followMenu(mon, BOX);
        to = g_read.atCenter, toYaw = g_read.atYaw, toPitch = g_read.atPitch;
        toScale = g_read.atScale;
    } else
        readingPose(mon, BOX, g_read.cinema, to, toYaw, toPitch, toScale);
    const float TO_DIM = g_read.cinema && !g_read.back ? kCinemaDim : 0.0f;
    g_read.dim = g_read.legDim + (TO_DIM - g_read.legDim) * P;

    const float TWO_PI = 2.f * std::numbers::pi_v<float>;
    E.center = g_read.legCenter + (to - g_read.legCenter) * P;
    E.yaw    = g_read.legYaw + std::remainder(toYaw - g_read.legYaw, TWO_PI) * P;
    E.pitch  = g_read.legPitch + (toPitch - g_read.legPitch) * P;
    E.roll   = g_read.legRoll + std::remainder(toRoll - g_read.legRoll, TWO_PI) * P;
    const float SCALE = g_read.legScale + (toScale - g_read.legScale) * P;
    E.width  = World3D::toWorld(BOX.w) * SCALE;
    E.height = World3D::toWorld(BOX.h) * SCALE;

    if (RAW < 1.0f)
        return;
    if (g_read.back)
        g_read = {}; // the room's scale takes it over on the next frame
    else if (!g_read.arrived) {
        g_read.arrived  = true;
        g_read.atCenter = to, g_read.atYaw = toYaw, g_read.atPitch = toPitch;
        g_read.atScale  = toScale;
        g_read.tick     = std::chrono::steady_clock::now();
    }
}

// F2 or F4 on the aimed window (or on `only`, the menu); the same key again
// sends it back, the other one turns reading into the cinema and back. The
// next leg starts from wherever the window is now.
static void toggleReading(bool cinema, std::uintptr_t only = 0) {
    const std::uintptr_t ID = only ? only : g_read.id ? g_read.id : g_lastAimedId;
    const auto* E = ID ? g_world.find(ID) : nullptr;
    if (!E) {
        notify(cinema ? "[hypr3d] F4: aim at a window to show it big" :
                        "[hypr3d] F2: aim at a window to read it",
               CHyprColor{0.2f, 0.8f, 0.4f, 1.0f});
        return;
    }
    if (!g_read.id) {
        g_read            = {};
        g_read.id         = ID;
        g_read.cinema     = cinema;
        g_read.roomCenter = E->center;
        g_read.roomYaw = E->yaw, g_read.roomPitch = E->pitch, g_read.roomRoll = E->roll;
    } else if (g_read.back) {
        g_read.back   = false;
        g_read.cinema = cinema;
    } else if (g_read.cinema != cinema)
        g_read.cinema = cinema;
    else
        g_read.back = true;
    g_read.legDim    = g_read.dim;
    g_read.arrived   = false;
    g_read.start     = std::chrono::steady_clock::now();
    g_read.legCenter = E->center;
    g_read.legYaw = E->yaw, g_read.legPitch = E->pitch, g_read.legRoll = E->roll;
    g_read.legScale  = E->logicalWidth > 0.f ?
        E->width / World3D::toWorld(E->logicalWidth) : configWindowScale();
    damageCurrentMonitor();
}

// Leaving, a fullscreen or a grab ends reading at once: the room pose back
// unless the window is being carried off.
static void endReading(bool putBack) {
    if (!g_read.id)
        return;
    if (auto* E = putBack ? g_world.find(g_read.id) : nullptr) {
        E->center = g_read.roomCenter;
        E->yaw = g_read.roomYaw, E->pitch = g_read.roomPitch, E->roll = g_read.roomRoll;
    }
    g_read = {};
    g_scene.setFeatured(0, 0.0f);
}

// The waste bin (config trash = { at, radius, height }): a window carried
// with Super+LMB and let go while the crosshair is on the bin falls in -- it
// tumbles and shrinks into the opening like a sheet crumpled up -- and is
// asked to close, as Super+Q asks. An application that stays (an unsaved
// file asks first) gets its window back after kTrashGrace. The bin's look is
// a scene object at the same place; the plugin knows only the zone.
static Vec3  g_trashAt{};
static float g_trashRadius = 0.0f, g_trashHeight = 0.0f; // radius 0: no bin
struct STrashed {
    std::uintptr_t id = 0;
    std::chrono::steady_clock::time_point start;
    Vec3  fromCenter{};
    float fromYaw = 0.f, fromPitch = 0.f, fromRoll = 0.f, fromScale = 1.f;
    bool  asked = false; // the close request went out
    std::chrono::steady_clock::time_point askedAt;
};
static std::vector<STrashed> g_trashed;
static constexpr float kTrashSeconds = 0.45f;
static constexpr float kTrashGrace   = 3.0f;

// The process gun (F7) -- "360 noscope discord when it freezes", four people
// in the Reddit finds. Aim at a window and shoot: a left click asks it to
// close, as Super+Q asks, and it jolts back where it was hit; hold the button
// kGunKillSeconds and its process is killed (SIGKILL), as Hyprland's
// forcekillactive does -- for a program that hangs, which Hyprland's own
// check (CANRManager) marks: the ring around the crosshair fills red. A
// program asked to close may still ask to save; only the held kill does not
// ask. F7 again or Escape puts it away; while it is out, the left button does
// not reach the windows.
static bool g_gun = false;
struct SGunShot {
    std::uintptr_t id = 0;
    std::chrono::steady_clock::time_point at;
    Vec3 push{}; // the shot's direction
};
static std::vector<SGunShot> g_gunShots; // windows still jolting back
struct SGunHold {
    bool           active = false;
    std::uintptr_t id = 0;
    std::chrono::steady_clock::time_point since;
};
static SGunHold        g_gunHold;
static void            gunTick(); // with the trigger, by onMouseButton
static constexpr float kGunKillSeconds  = 1.0f;
static constexpr float kGunJoltSeconds  = 0.3f;
static constexpr float kGunJolt         = 0.25f; // m back at the hit

// A window hit a moment ago is pushed back along the shot and springs home.
static void applyGunJolt(std::uintptr_t id, World3D::SEntity& E) {
    const auto NOW = std::chrono::steady_clock::now();
    for (const auto& S : g_gunShots) {
        if (S.id != id)
            continue;
        const float T = std::chrono::duration<float>(NOW - S.at).count() / kGunJoltSeconds;
        if (T < 1.0f)
            E.center = E.center + S.push * (kGunJolt * (1.0f - T) * (1.0f - T));
    }
}

// The TV (config tv = { at = {x,y,z}, yaw = deg, width, height }): a screen
// in the world a window can be put on -- carried with Super+LMB and let go
// on it, the window snaps onto the picture, fitted, and stays; carried off
// again, it is free. Super+wheel on it scales the window itself: its
// resolution changes, the picture keeps its size, so the text grows or
// shrinks as on a real TV. Pinned, it is resized to the screen's shape.
// `at` is the picture's centre, `yaw` the yaw a window on it gets (180: it
// faces -z), width and height the picture's in metres; width 0: no TV.
static Vec3           g_tvAt{};
static float          g_tvYawDeg = 0.0f, g_tvW = 0.0f, g_tvH = 0.0f;
static std::uintptr_t g_tvWindowId = 0;

static Vec3 tvNormal() {
    const float Y = g_tvYawDeg * std::numbers::pi_v<float> / 180.0f;
    return Vec3{std::sin(Y), 0.f, std::cos(Y)};
}

// The crosshair's ray on the TV's picture.
static bool aimAtTV() {
    if (g_tvW <= 0.0f || g_tvH <= 0.0f)
        return false;
    const auto& CAM = g_scene.camera();
    const Vec3  O = CAM.position, D = CAM.centerRay(), N = tvNormal();
    const float DN = dot(D, N);
    if (std::fabs(DN) < 1e-4f)
        return false;
    const float T = dot(g_tvAt - O, N) / DN;
    if (T <= 0.0f || T > 12.0f)
        return false;
    const Vec3 R{N.z, 0.f, -N.x}; // the picture's right, seen from in front
    const Vec3 P = O + D * T - g_tvAt;
    return std::fabs(dot(P, R)) <= g_tvW * 0.5f + 0.05f && std::fabs(P.y) <= g_tvH * 0.5f + 0.05f;
}

// Resizes a window's 2D box about its centre: the TV's shape, `height` px.
static void tvResize(const PHLWINDOW& W, double height) {
    const auto   BOX = Compat::currentWindowBox(W);
    const double H = std::clamp(height, 360.0, 2160.0);
    const double WD = std::round(H * g_tvW / g_tvH);
    Compat::setWindowBox(W, CBox{BOX.x + (BOX.w - WD) / 2, BOX.y + (BOX.h - H) / 2, WD, std::round(H)});
}

// A carried window let go on the TV: it is put on it.
static bool dropOnTV(std::uintptr_t id) {
    const auto W = id ? Compat::findWindowById(id) : nullptr;
    if (!W || !aimAtTV())
        return false;
    if (g_read.id == id)
        endReading(false);
    g_tvWindowId = id;
    tvResize(W, Compat::currentWindowBox(W).h);
    return true;
}

// One window of syncWorld's loop on the TV: fitted onto the picture, a
// millimetre in front of the glass.
static void applyTV(std::uintptr_t id, const CBox& BOX, World3D::SEntity& E) {
    if (id != g_tvWindowId || g_tvW <= 0.0f || BOX.w <= 0 || BOX.h <= 0)
        return;
    const float BW = World3D::toWorld(BOX.w), BH = World3D::toWorld(BOX.h);
    const float FIT = std::min(g_tvW / BW, g_tvH / BH);
    E.center = g_tvAt + tvNormal() * 0.002f;
    E.yaw    = g_tvYawDeg * std::numbers::pi_v<float> / 180.0f;
    E.pitch = E.roll = 0.0f;
    E.width  = BW * FIT;
    E.height = BH * FIT;
}

// The crosshair's ray on the bin: its side within its height, or its opening.
static bool aimAtBin() {
    if (g_trashRadius <= 0.0f)
        return false;
    const auto& CAM = g_scene.camera();
    const Vec3  O = CAM.position, D = CAM.centerRay();
    const float TOP = g_trashAt.y + g_trashHeight;
    const auto  IN_HEIGHT = [&](float t) {
        const float Y = O.y + t * D.y;
        return t > 0.0f && t < 12.0f && Y >= g_trashAt.y && Y <= TOP;
    };
    const float FX = O.x - g_trashAt.x, FZ = O.z - g_trashAt.z;
    const float A = D.x * D.x + D.z * D.z;
    const float B = 2.0f * (FX * D.x + FZ * D.z);
    const float C = FX * FX + FZ * FZ - g_trashRadius * g_trashRadius;
    const float DISC = B * B - 4.0f * A * C;
    if (A > 1e-6f && DISC >= 0.0f) {
        const float SQ = std::sqrt(DISC);
        if (IN_HEIGHT((-B - SQ) / (2.0f * A)) || IN_HEIGHT((-B + SQ) / (2.0f * A)))
            return true;
    }
    if (std::fabs(D.y) > 1e-6f) {
        const float T = (TOP - O.y) / D.y;
        const float X = O.x + T * D.x - g_trashAt.x, Z = O.z + T * D.z - g_trashAt.z;
        if (T > 0.0f && T < 12.0f && X * X + Z * Z <= g_trashRadius * g_trashRadius)
            return true;
    }
    return false;
}

// Let go of a carried window on the bin: it starts falling in.
static bool dropInBin(std::uintptr_t id) {
    const auto* E = id ? g_world.find(id) : nullptr;
    if (!E || !Compat::findWindowById(id) || !aimAtBin())
        return false;
    if (std::ranges::any_of(g_trashed, [&](const STrashed& t) { return t.id == id; }))
        return true;
    if (g_read.id == id)
        endReading(false);
    STrashed T;
    T.id = id;
    T.start = std::chrono::steady_clock::now();
    T.fromCenter = E->center;
    T.fromYaw = E->yaw, T.fromPitch = E->pitch, T.fromRoll = E->roll;
    T.fromScale = E->logicalWidth > 0.f ? E->width / World3D::toWorld(E->logicalWidth) : configWindowScale();
    g_trashed.push_back(T);
    return true;
}

// One window of syncWorld's loop on its way into the bin, or waiting there
// for its application to close it. False: it stayed open -- back it comes.
static bool applyTrash(STrashed& T, const CBox& BOX, World3D::SEntity& E) {
    const auto NOW = std::chrono::steady_clock::now();
    const float RAW = std::clamp(std::chrono::duration<float>(NOW - T.start).count() / kTrashSeconds, 0.0f, 1.0f);
    const float P = RAW * RAW; // falling: slow, then fast
    // Over the opening, then down into it: an arc that drops in at the end.
    const Vec3 OVER = g_trashAt + Vec3{0.f, g_trashHeight + 0.35f, 0.f};
    const Vec3 IN   = g_trashAt + Vec3{0.f, g_trashHeight * 0.5f, 0.f};
    const Vec3 MID  = T.fromCenter + (OVER - T.fromCenter) * P;
    E.center = MID + (IN - MID) * (P * P);
    E.yaw    = T.fromYaw + 3.0f * P;   // tumbling as it crumples
    E.pitch  = T.fromPitch + 2.2f * P;
    E.roll   = T.fromRoll + 4.0f * P;
    const float SCALE = T.fromScale * (1.0f - 0.985f * P);
    E.width  = World3D::toWorld(BOX.w) * SCALE;
    E.height = World3D::toWorld(BOX.h) * SCALE;

    if (RAW >= 1.0f && !T.asked) {
        T.asked = true;
        T.askedAt = NOW;
        // Outside the frame: render.stage is no place to talk to clients.
        const std::uintptr_t ID = T.id;
        if (g_pEventLoopManager)
            g_pEventLoopManager->doLater([ID] {
                if (const auto W = Compat::findWindowById(ID))
                    W->sendClose();
            });
    }
    if (T.asked && std::chrono::duration<float>(NOW - T.askedAt).count() > kTrashGrace) {
        E.center = T.fromCenter;
        E.yaw = T.fromYaw, E.pitch = T.fromPitch, E.roll = T.fromRoll;
        E.width  = World3D::toWorld(BOX.w) * T.fromScale;
        E.height = World3D::toWorld(BOX.h) * T.fromScale;
        return false;
    }
    return true;
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
    const auto INFOS = eligibleWindowsSpanned();

    // A grab takes the window being read where the hand goes; a fullscreen
    // takes it to the screen. Either way it is not read any more.
    if (g_read.id && ((g_pointerDown && g_pointerGesture != EPointerGesture::None) ||
                      g_fsPhase != EFullscreenPhase::None))
        endReading(false);

    g_winOutlines.clear();

    const float MONW = mon->m_size.x;
    const float MONH = mon->m_size.y;

    // Built afresh (nothing in the room yet): how large the wall may be.
    if (g_cfgSpan && g_wallFit <= 0.0f && g_world.entities().empty()) {
        const auto PLANE0 = viewPlane();
        const auto MONS   = spannedMonitors();
        if (PLANE0 && MONS.size() > 1) {
            const auto& CAM = g_scene.camera();
            const Vec3  FWD = CAM.forward();
            const Vec3  UP  = cross(CAM.right(), FWD);
            // A wall point's height over the eye, per world unit per px.
            double lowest = 0.0;
            for (const auto& M : MONS)
                lowest = std::min(lowest, FWD.y * PLANE0->distance +
                                              UP.y * (PLANE0->cy - (M->m_position.y + M->m_size.y)));
            g_wallFit = lowest < 0.0 ?
                static_cast<float>((Camera::kEyeHeight - kWallFloorGap) / -lowest / World3D::toWorld(1.0f)) :
                g_cfgWindowScale;
        }
    }

    const float WIN_SCALE = configWindowScale();

    std::vector<World3D::SEntity> ENTITIES;
    ENTITIES.reserve(INFOS.size());

    // Windows new to the room in this rebuild -- all of them when the view
    // is turned on -- stand side by side on an arc around the camera at the
    // spawn distance, in the order the 2D layout had them, left to right
    // (then top to bottom). All at the one spawn point, they covered each
    // other and the room showed a single panel. One alone still spawns
    // straight ahead. Each slot's angle is the arc length of the windows
    // before it over the radius, with a gap between neighbours.
    // Spanning several monitors, new windows go on the wall (below), not on
    // the arc.
    const auto PLANE = viewPlane();
    const bool WALL  = g_cfgSpan && PLANE && spannedMonitors().size() > 1;

    // On the wall, windows that overlap on the desk would lie in one plane and
    // cut into each other: each steps toward the eye once per window it covers
    // -- panels over floating windows over tiled ones over bottom-layer panels,
    // a later one over an earlier (Hyprland lists windows bottom to top) -- one
    // slab thickness and a gap per step. Tiles side by side stay on the wall.
    std::unordered_map<std::uintptr_t, int> wallStack;
    if (WALL) {
        std::vector<const Compat::SWindowInfo*> ORDER;
        ORDER.reserve(INFOS.size());
        for (const auto& info : INFOS)
            ORDER.push_back(&info);

        const auto CLASS = [](const Compat::SWindowInfo* i) {
            if (i->isLayer)
                return i->layer && i->layer->m_layer <= 1 ? 0 : 3;
            return i->floating ? 2 : 1;
        };
        std::stable_sort(ORDER.begin(), ORDER.end(),
                         [&](const auto* a, const auto* b) { return CLASS(a) < CLASS(b); });

        const auto RECT = [](const Compat::SWindowInfo* i) {
            CBox r = i->monitorLocalBox;
            if (i->monitor) {
                r.x += i->monitor->m_position.x;
                r.y += i->monitor->m_position.y;
            }
            return r;
        };

        for (size_t i = 0; i < ORDER.size(); ++i) {
            int step = 0;
            for (size_t j = 0; j < i; ++j)
                if (RECT(ORDER[i]).overlaps(RECT(ORDER[j])))
                    step = std::max(step, wallStack[ORDER[j]->id] + 1);
            wallStack[ORDER[i]->id] = step;
        }
    }

    // A surface the room remembers is not new: it goes back where it stood
    // (see g_room), and takes no slot on the arc.
    const auto REMEMBERED = [](const Compat::SWindowInfo& info) -> const SRememberedPose* {
        const auto IT = g_room.poses.find(info.id);
        if (IT == g_room.poses.end())
            return nullptr;
        const bool SAME = info.isLayer ? IT->second.layer.lock() == info.layer
                                       : IT->second.window.lock() == info.window;
        return SAME ? &IT->second : nullptr;
    };

    std::unordered_map<std::uintptr_t, float> freshAngle;
    if (!WALL) {
        std::vector<const Compat::SWindowInfo*> FRESH;
        for (const auto& info : INFOS)
            if (!info.isLayer && !g_world.find(info.id) && !REMEMBERED(info))
                FRESH.push_back(&info);

        if (FRESH.size() > 1) {
            std::sort(FRESH.begin(), FRESH.end(), [](const auto* a, const auto* b) {
                if (a->monitorLocalBox.x != b->monitorLocalBox.x)
                    return a->monitorLocalBox.x < b->monitorLocalBox.x;
                return a->monitorLocalBox.y < b->monitorLocalBox.y;
            });

            const float RADIUS = std::max(g_cfgSpawnDistance, 0.5f);
            const float GAP    = World3D::toWorld(48.0f) * WIN_SCALE;
            float       total  = GAP * static_cast<float>(FRESH.size() - 1);
            for (const auto* f : FRESH)
                total += World3D::toWorld(f->monitorLocalBox.w) * WIN_SCALE;

            float along = -total * 0.5f;
            for (const auto* f : FRESH) {
                const float W = World3D::toWorld(f->monitorLocalBox.w) * WIN_SCALE;
                freshAngle[f->id] = (along + W * 0.5f) / RADIUS;
                along += W + GAP;
            }
        }
    }

    g_hudItems.clear();
    for (const auto& info : INFOS) {
        if (g_cfgHud && info.isLayer && info.layer && info.layer->m_layer >= 2) {
            g_hudItems.push_back({info.id, info.monitor, info.monitorLocalBox});
            continue;
        }
        // Track each window's last stable (non-transition) box -- see
        // g_fsStableBoxes. Skipped while the exit re-assert is running: a
        // late Hyprland-side restore could pollute the memory with the
        // monitor-sized box, and the NEXT fullscreen cycle would then
        // restore the giant size (the intermittent bug).
        if (!info.isLayer && info.window &&
            g_fsPhase == EFullscreenPhase::None && g_fsAssertFrames == 0 &&
            info.id != g_fsCurrentId) // a fullscreened window's box is the
                                      // monitor -- transient, never "stable"
            g_fsStableBoxes[info.id] = Compat::currentWindowBox(info.window);

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

        // Depth-slab FALLBACK silhouette for WINDOWS: an analytic rounded
        // rectangle of the decorated box -- the corner radius the compositor
        // reports plus the border inset. Used only when the texture-traced
        // outline is unavailable (the GPU copy failed or found no shape);
        // the primary source lives in the capture layer.
        if (!info.isLayer && info.window) {
            const float R = std::max(
                0.0f, info.window->rounding() +
                          static_cast<float>(SNAPSHOT->surfaceOffset.x));

            auto& C = g_winOutlines[info.id];

            if (!C.loops || C.w != static_cast<int>(BOX.w) ||
                C.h != static_cast<int>(BOX.h) || C.r != static_cast<int>(R)) {
                C.w = static_cast<int>(BOX.w);
                C.h = static_cast<int>(BOX.h);
                C.r = static_cast<int>(R);

                SOutlineLoop LOOP =
                    roundedRectLoop(BOX.w, BOX.h, R, 3.0f, 10);

                C.loops = std::make_shared<const std::vector<SOutlineLoop>>(
                    LOOP.pts.size() >= 3
                        ? std::vector<SOutlineLoop>{std::move(LOOP)}
                        : std::vector<SOutlineLoop>{});
            }
        }

        // Uniform window scale from config, applied to every entity every
        // frame so a runtime change resizes the whole room. The scale is
        // uniform so the captured content never distorts.
        entity.spawnScale = WIN_SCALE;

        // Seed pose: existing entities own their world position and rotation.
        // NEW windows spawn straight in front of the camera at a fixed read
        // distance, facing it.
        if (const auto* EXISTING = g_world.find(info.id)) {
            entity.center = EXISTING->center;
            entity.yaw = EXISTING->yaw;
            entity.pitch = EXISTING->pitch;
            entity.roll = EXISTING->roll;
        }
        else if (const auto* MEM = REMEMBERED(info)) {
            entity.center = MEM->center;
            entity.yaw    = MEM->yaw;
            entity.pitch  = MEM->pitch;
            entity.roll   = MEM->roll;
        }
        else {
            const auto& CAM = g_scene.camera();
            Vec3 FWD = CAM.forward();

            if (WALL && info.monitor) {
                // Spanning several monitors the desktop becomes a wall: each
                // window where it stood, on the view plane at the distance at
                // which one of its pixels covers one pixel of the screen. On
                // every monitor the room then opens on exactly the desktop it
                // replaces. (On a sphere around the eye a side screen's
                // windows overlapped: 66 degrees off the axis, a flat screen
                // spans a quarter of the angle it would face-on.)
                const double K  = World3D::toWorld(1.0f) * WIN_SCALE; // world units per logical px
                const double GX = info.monitor->m_position.x + info.monitorLocalBox.x +
                    info.monitorLocalBox.w * 0.5;
                const double GY = info.monitor->m_position.y + info.monitorLocalBox.y +
                    info.monitorLocalBox.h * 0.5;
                const Vec3 RIGHT = CAM.right();
                const Vec3 UP    = cross(RIGHT, FWD);

                const float STEP = std::max(g_cfgWindowDepth, 0.0f) + 0.02f;
                const auto  RANK = wallStack.find(info.id);
                const float NEARER =
                    RANK != wallStack.end() ? STEP * static_cast<float>(RANK->second) : 0.0f;

                // Toward the eye along its own sight line, not the view axis:
                // stepped along the axis, a panel far off it projected further
                // out -- a side screen's bar slid past that screen's edge.
                const Vec3 ONWALL =
                    FWD * static_cast<float>(PLANE->distance * K) +
                    RIGHT * static_cast<float>((GX - PLANE->cx) * K) +
                    UP * static_cast<float>((PLANE->cy - GY) * K);
                const float LEN = std::sqrt(dot(ONWALL, ONWALL));

                entity.center = CAM.position +
                    ONWALL * (LEN > NEARER ? (LEN - NEARER) / LEN : 1.0f);
            }
            else {
                // Its slot on the arc (see freshAngle): turned about the
                // camera's up, to the right for a positive angle.
                if (const auto SLOT = freshAngle.find(info.id); SLOT != freshAngle.end()) {
                    const Vec3 RIGHT = CAM.right();
                    FWD = normalize(FWD * std::cos(SLOT->second) + RIGHT * std::sin(SLOT->second));
                }

                entity.center = CAM.position + FWD * g_cfgSpawnDistance;
            }

            // Face the camera: with this model's convention (the normal's Y
            // component is -sin(pitch)) the target is the camera's own yaw
            // and pitch.
            entity.yaw   = std::atan2(-FWD.x, -FWD.z);
            entity.pitch = std::asin(std::clamp(FWD.y, -1.0f, 1.0f));
        }

        // The quad size ALWAYS follows the snapshot box (times the spawn
        // scale): the UV subrect and the input mapping are box-relative, so a
        // stale world size would squish the content and shrink the input zone
        // on every resize.
        entity.width  = World3D::toWorld(BOX.w) * entity.spawnScale;
        entity.height = World3D::toWorld(BOX.h) * entity.spawnScale;

        if (g_read.id == info.id)
            applyReading(mon, BOX, entity);
        if (const auto T = std::ranges::find_if(g_trashed, [&](const STrashed& t) { return t.id == info.id; });
            T != g_trashed.end() && !applyTrash(*T, BOX, entity)) {
            if (const auto W = Compat::findWindowById(T->id))
                notify("[hypr3d] " + W->m_title + " stayed open; it is back", CHyprColor{1.0f, 0.6f, 0.2f, 1.0f});
            g_trashed.erase(T);
        }
        applyGunJolt(info.id, entity);
        applyTV(info.id, BOX, entity);

        // Fullscreen transition: the animation owns this quad's size, and it
        // MUST be applied here -- the draw list below is built from these
        // values, so overriding later in applyFullscreenAnimation only
        // reaches the render a frame late. That late frame is what leaked
        // the config scale into the monitor-facing handoff and left a seam.
        // The scale glides between the room's config scale and the screen's
        // exact 1:1, following the same eased progress as the box lerp.
        if (g_fsPhase != EFullscreenPhase::None) {
            if (auto FSW = g_fsWindow.lock();
                FSW && Compat::windowId(FSW) == info.id) {
                const float RAWP = fsRawProgress();
                const float p = RAWP * RAWP * (3.0f - 2.0f * RAWP);
                const float ROOM_S = configWindowScale();
                const float S0 =
                    g_fsPhase == EFullscreenPhase::To2D ? ROOM_S : 1.0f;
                const float S1 =
                    g_fsPhase == EFullscreenPhase::To2D ? 1.0f : ROOM_S;
                const float S = S0 + (S1 - S0) * p;

                entity.width  = World3D::toWorld(BOX.w) * S;
                entity.height = World3D::toWorld(BOX.h) * S;
            }
        }

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

    if (g_read.id && std::ranges::none_of(ENTITIES, [](const auto& E) { return E.id == g_read.id; }))
        g_read = {}; // it closed
    if (g_tvWindowId && std::ranges::none_of(ENTITIES, [](const auto& E) { return E.id == g_tvWindowId; }))
        g_tvWindowId = 0; // it closed: its address can come back for a new window
    std::erase_if(g_trashed, [&](const STrashed& t) {
        return std::ranges::none_of(ENTITIES, [&](const auto& E) { return E.id == t.id; });
    }); // closed: in the bin for good

    // The menu window, as it opens, comes to the eye at 1:1 -- over a window
    // being read, which goes back first.
    std::uintptr_t menu = 0;
    if (!g_cfgMenuTitle.empty())
        for (const auto& E : ENTITIES)
            if (const auto W = Compat::findWindowById(E.id); W && W->m_title == g_cfgMenuTitle)
                menu = E.id;
    const bool MENU_NEW = menu && menu != g_menuId;
    g_menuId = menu;
    g_world.setEntities(std::move(ENTITIES), false);
    updateShadows();
    if (MENU_NEW) {
        if (g_read.id && g_read.id != menu)
            endReading(true);
        if (!g_read.id)
            toggleReading(false, menu);
    }
    g_scene.setFeatured(g_read.id, g_read.dim);

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
        render.roll = entity.roll;

        render.width  = entity.width;
        render.height = entity.height;

        // The room fade (see g_fsFade): every non-FS window rides it. The
        // fullscreen window must NEVER ride it, in any of its three
        // identities: the one fullscreened right now (g_fsCurrentId), the
        // one being animated back to the room (To3D -- FSW is already null
        // there, but g_fsWindow still holds it), and the one that JUST
        // exited (g_fsLastFSWindow persists past the exit -- without this
        // the ex-FS panel fades in together with everyone else even though
        // it was never hidden and must sit at 100% immediately).
        const auto FS_WIN      = g_fsWindow.lock();
        const auto LAST_FS_WIN = g_fsLastFSWindow.lock();

        const bool IS_FS_WINDOW =
            entity.id == g_fsCurrentId ||
            (FS_WIN && entity.id == Compat::windowId(FS_WIN)) ||
            (LAST_FS_WIN && entity.id == Compat::windowId(LAST_FS_WIN));

        render.alpha = IS_FS_WINDOW ? 1.0f : g_fsFade;

        // Depth slab (windows.depth): silhouette source priority -- the
        // outline traced from the snapshot texture's real alpha (exact
        // corner shape; the GPU-side copy is the only reliable read), then
        // the analytic rounded rect (mask refresh failures, first frames),
        // then GLScene's plain box.
        render.depth = g_cfgWindowDepth;

        if (SNAPSHOT->outlines && !SNAPSHOT->outlines->empty())
            render.outlines = SNAPSHOT->outlines;
        else if (auto IT = g_winOutlines.find(entity.id);
                 IT != g_winOutlines.end() && IT->second.loops &&
                 !IT->second.loops->empty())
            render.outlines = IT->second.loops;

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

    // Aim focus updates freeze during the fullscreen transition: the flying
    // quad sweeps the crosshair across other windows and would thrash focus.
    if (g_fsPhase == EFullscreenPhase::None)
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

// Whether the pixel a ray hit shows anything: empty pixels of an overlay let
// the ray through to the surface underneath.
static bool hitVisible(const World3D::SHit& HIT) {
    const auto* SNAPSHOT = g_capture.get(HIT.id);

    if (SNAPSHOT && SNAPSHOT->alphaValid && !SNAPSHOT->alphaMask.empty()) {
        const int MX = std::min(SNAPSHOT->alphaW - 1,
            static_cast<int>(HIT.u * SNAPSHOT->alphaW));
        const int MY = std::min(SNAPSHOT->alphaH - 1,
            static_cast<int>(HIT.v * SNAPSHOT->alphaH));

        return SNAPSHOT->alphaMask[static_cast<size_t>(MY) * SNAPSHOT->alphaW + MX] >= 8;
    }
    return true;
}

static World3D::SHit aimHit() {
    const auto& cam = g_scene.camera();

    // Nearest hit whose pixels are actually visible: an invisible layer
    // overlay (transparent quickshell PanelWindow) must not swallow the
    // crosshair -- empty pixels fall through to the surface underneath. This
    // is also what keeps aim/focus from flapping between an overlay and the
    // window it covers.
    for (const auto& HIT : g_world.pickAll(cam.position, cam.centerRay())) {
        if (!hitVisible(HIT))
            continue; // transparent pixel of an overlay: next surface
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

    // The quad spans toWorld(box) * spawnScale world units while carrying
    // box pixels of content, so the px density on the quad is the room's
    // base density divided by the scale.
    const auto* RESIZE_ENTITY = g_world.find(g_resize.id);
    const double scale =
        RESIZE_ENTITY ? RESIZE_ENTITY->spawnScale : 1.0f;

    const double currentCenterX = CURRENT.x + CURRENT.w * 0.5;
    const double currentCenterY = CURRENT.y + CURRENT.h * 0.5;

    return {
        currentCenterX + static_cast<double>(LOCAL.x) *
            World3D::LOGICAL_PX_PER_UNIT / scale,
        currentCenterY - static_cast<double>(LOCAL.y) *
            World3D::LOGICAL_PX_PER_UNIT / scale,
    };
}

static CBox resizeBoxFromAim(const Vec3& point, const PHLMONITOR& mon) {
    // The grabbed edge moves by the crosshair's travel since the grab, not
    // to its absolute position: the box starts identical to the window, so
    // starting a resize never snaps the nearest corner under the crosshair.
    const Vector2D PX = worldPointToGlobalPx(mon, point);
    const double DX = PX.x - g_resize.grabPx.x;
    const double DY = PX.y - g_resize.grabPx.y;

    const CBox& start = g_resize.startBox;
    const double RIGHT = start.x + start.w;
    const double BOTTOM = start.y + start.h;

    CBox out = start;

    if (g_resize.edgeX > 0) {
        out.x = start.x;
        out.w = start.w + DX;
    } else {
        out.x = start.x + DX;
        out.w = start.w - DX;
    }

    if (g_resize.edgeY > 0) {
        out.y = start.y + DY;
        out.h = start.h - DY;
    } else {
        out.y = start.y;
        out.h = start.h + DY;
    }

    const auto MIN = g_resize.window->minSize().value_or(Vector2D{1.0, 1.0});
    const auto MAX = g_resize.window->maxSize().value_or(Vector2D{INFINITY, INFINITY});

    out.w = std::clamp(out.w, MIN.x, MAX.x);
    out.h = std::clamp(out.h, MIN.y, MAX.y);

    if (g_resize.edgeX > 0)
        out.x = start.x;
    else
        out.x = RIGHT - out.w;

    if (g_resize.edgeY > 0)
        out.y = BOTTOM - out.h;
    else
        out.y = start.y;

    return out;
}

// --- fullscreen passthrough -------------------------------------------------

static float wrapPi(float a) {
    while (a > 3.14159265f)
        a -= 6.28318530f;
    while (a < -3.14159265f)
        a += 6.28318530f;
    return a;
}

// To2D: capture the room pose, force the real window to the monitor box and
// aim the animation at a quad that exactly covers the frozen camera frustum.
static void startTo2D(const PHLWINDOW& window, bool captureRestoreBox) {
    const auto MON = targetMonitor();

    if (!MON)
        return;

    const auto ID = Compat::windowId(window);

    if (!g_world.find(ID))
        return; // not in the room: nothing to animate

    g_fsWindow = window;

    // The floating box: the POSITION from the stable memory (where the
    // window lived), the SIZE from Hyprland's own remembered floating size.
    // When the FS was engaged, Hyprland remembered the pre-fullscreen size
    // there (setTargetFullscreenModeInternal does it BEFORE stretching the
    // window to the monitor) -- exactly the size the 2D FS-exit restores.
    // The stable memory is the fallback (e.g. a tiled window has no
    // floating history: its tile size is the honest original).
    if (captureRestoreBox) {
        const auto IT = g_fsStableBoxes.find(ID);
        CBox RESTORE = IT != g_fsStableBoxes.end() ?
            IT->second : Compat::currentWindowBox(window);

        if (window->m_target) {
            const auto LF = window->m_target->lastFloatingSize();

            if (LF.x > 5.0 && LF.y > 5.0)
                RESTORE = CBox{RESTORE.x, RESTORE.y, LF.x, LF.y};
        }

        g_fsRestoreBox = RESTORE;
    }

    // The fullscreen TARGET box (the animation moves the real box there
    // gradually -- see applyFullscreenAnimation).
    g_fsMonitorBox = CBox{
        MON->m_position.x, MON->m_position.y, MON->m_size.x, MON->m_size.y};

    // Pin Hyprland's remembered floating size to the captured PRE-FS size:
    // the transition's per-frame setWindowBox calls would otherwise
    // re-member intermediate (up to monitor-sized) values, and Hyprland's
    // own FS-exit applies that memory.
    if (window->m_target)
        window->m_target->rememberFloatingSize(
            {g_fsRestoreBox.w, g_fsRestoreBox.h});

    // The screen-covering distance derives from the vertical FOV and the
    // monitor's NATURAL world size (logical px / 100):
    // d = (worldHeight / 2) / tan(vfov / 2). At that distance a
    // monitor-sized quad subtends exactly the full frustum.
    const float MON_H_WORLD =
        static_cast<float>(MON->m_size.y) / World3D::LOGICAL_PX_PER_UNIT;
    g_fsDistance = (MON_H_WORLD * 0.5f) /
        std::tan(kFovDeg * 3.14159265f / 360.0f);

    if (const auto* E = g_world.find(ID)) {
        g_fsStartCenter = E->center;
        g_fsStartYaw    = E->yaw;
        g_fsStartPitch  = E->pitch;
        g_fsStartRoll   = wrapPi(E->roll);
        g_fsSavedRoll   = wrapPi(E->roll); // restored when the FS exits

    }

    const auto& CAM = g_scene.camera();
    const Vec3 FWD = CAM.forward();

    g_fsEndCenter = CAM.position + FWD * g_fsDistance;

    // Face the FROZEN camera: with the model convention (normal Y is
    // -sin(pitch)) facing the camera means yaw = -camYaw, pitch = +camPitch.
    // The content lands unmirrored: the quad's local +X maps exactly onto
    // the camera's right vector.
    g_fsEndYaw    = -CAM.yaw;
    g_fsEndPitch  = CAM.pitch;

    resetPointerGesture(); // no gestures on the animating window
    g_input.reset();       // no look jump when look resumes

    g_fsPhase        = EFullscreenPhase::To2D;
    g_fsPhaseStart   = std::chrono::steady_clock::now();
    g_fsWasOn        = true;
    g_fsAssertFrames = 0;
    damageCurrentMonitor();
}

// In2D -> To3D: restore the real floating box, capture input, hide the
// cursor; the animation runs the quad back to its room pose.
static void startTo3D() {
    if (auto W = g_fsWindow.lock()) {
        // The To3D start is the window's CURRENT pose: the tracked fullscreen
        // pose after a normal exit, or wherever the aborted To2D got to. The
        // real floating box is restored when the animation COMPLETES (restoring
        // it here would make the quad show magnified small-window content).
        if (auto* E = g_world.find(Compat::windowId(W))) {
            g_fsEndCenter = E->center;
            g_fsEndYaw    = E->yaw;
            g_fsEndPitch  = E->pitch;
            g_fsRollAtStart = wrapPi(E->roll);
        }

        if (Compat::setCursorHidden(true))
            Pointer::mgr()->resetCursorImage();

        Compat::setPointerCapture(g_hookInstalled);

        g_fsPhase      = EFullscreenPhase::To3D;
        g_fsPhaseStart = std::chrono::steady_clock::now();
        g_input.reset();
        damageCurrentMonitor();
    }
}

// None -> To2D when a fullscreen window appears; In2D -> To3D when it goes.
// Hyprland fades windows with per-window alpha animations that only make
// progress while the window is being damaged: during the fullscreen
// handoff the non-FS windows fade out (their FADE/FULLSCREEN alpha channels
// animate toward 0), and on the way back they fade in; focus changes fade
// the ACTIVE channel the same way. Mid-fade the compositor stops damaging
// them -- the animation freezes a couple of frames in, and the room shows
// windows stuck at partial alpha until an aim hover forces a frame (the
// semi-transparent-after-FS bug). Damage every window whose alpha channels
// are still in flight; the damage drives the fade to completion and the
// loop stops on its own.
static void damageWindowsWithLiveAlpha(const PHLMONITOR& mon) {
    if (!Desktop::windowState() || !mon || !mon->m_activeWorkspace)
        return;

    for (const auto& W : Desktop::windowState()->windows()) {
        if (!W || W->m_workspace != mon->m_activeWorkspace)
            continue;

        const auto IN_FLIGHT = [&](Desktop::View::eWindowAlpha channel) {
            return W->alphaValue(channel) != W->alphaGoal(channel);
        };

        if (IN_FLIGHT(Desktop::View::WINDOW_ALPHA_FADE) ||
            IN_FLIGHT(Desktop::View::WINDOW_ALPHA_ACTIVE) ||
            IN_FLIGHT(Desktop::View::WINDOW_ALPHA_FULLSCREEN))
            g_pHyprRenderer->damageWindow(W);
    }
}

static void pollFullscreen() {
    const auto MON = targetMonitor();

    if (!MON)
        return;

    // The alpha keepalive rides the pump (8 ms) rather than the render pass,
    // so the fade keeps ticking even when a frame is slow.
    damageWindowsWithLiveAlpha(MON);

    const auto FSW = Fullscreen::controller()->getFullscreenWindow(MON);

    g_fsCurrentId = FSW ? Compat::windowId(FSW) : 0;

    // Our own room fade: while a fullscreen exists (or a transition runs)
    // the other windows glide to invisible; when it is gone they glide
    // back. Deterministic wall-clock ticks -- never dependent on Hyprland's
    // animation engine or on snapshot retakes.
    constexpr float FADE_DURATION = 0.25f;

    const auto FADE_NOW = std::chrono::steady_clock::now();
    const float FADE_DT = g_fsFadeLast.time_since_epoch().count() == 0 ?
        0.0f :
        std::chrono::duration<float>(FADE_NOW - g_fsFadeLast).count();
    g_fsFadeLast = FADE_NOW;

    const float FADE_TARGET =
        (FSW || g_fsPhase != EFullscreenPhase::None) ? 0.0f : 1.0f;

    if (FADE_DT > 0.0f && g_fsFade != FADE_TARGET) {
        const float STEP = std::min(FADE_DT / FADE_DURATION, 1.0f);
        g_fsFade = FADE_TARGET > g_fsFade ?
            std::min(g_fsFade + STEP, FADE_TARGET) :
            std::max(g_fsFade - STEP, FADE_TARGET);
    }

    // Super+F on a window whose 2D home is another monitor fullscreened it
    // THERE: the room went on drawing over that monitor, and the one in front
    // of the player showed nothing new (measured on the owner's three
    // monitors). A window that turns fullscreen elsewhere while the room is
    // open is brought to the workspace in front; the passthrough below takes
    // it on the next poll, and the end of the passthrough sends it home.
    std::vector<std::uintptr_t> elsewhere;
    PHLWINDOW bring;
    for (const auto& OTHER : spannedMonitors()) {
        if (OTHER == MON)
            continue;
        const auto W = Fullscreen::controller()->getFullscreenWindow(OTHER);
        if (!W)
            continue;
        const auto ID = Compat::windowId(W);
        elsewhere.push_back(ID);
        if (g_fsElsewhereKnown && !bring && g_world.find(ID) &&
            std::find(g_fsElsewhere.begin(), g_fsElsewhere.end(), ID) == g_fsElsewhere.end())
            bring = W;
    }
    g_fsElsewhere      = elsewhere;
    g_fsElsewhereKnown = true;
    if (bring && !FSW && g_fsPhase == EFullscreenPhase::None && ownsInput() &&
        MON->m_activeWorkspace) {
        g_fsHomeWindow    = bring;
        g_fsHomeWorkspace = bring->m_workspace;
        Config::Actions::moveToWorkspace(MON->m_activeWorkspace, true, bring);
        return;
    }

    if (g_fsPhase == EFullscreenPhase::None) {
        if (FSW && !g_fsWasOn && ownsInput())
            startTo2D(FSW);
        else {
            g_fsWasOn = FSW != nullptr;

            // A fullscreen EXITED while the room is plain 3D (no transition
            // was running -- e.g. the FS predated the 3D entry): make the
            // window a spawn-sized floating panel AND hold it with the same
            // condition-driven assert the To3D completion uses -- Hyprland's
            // floating layout re-applies ITS remembered size (possibly
            // monitor-sized from an old cycle), and a one-shot set loses to
            // it.
            if (!FSW && g_fsWasOn) {
                if (auto OLDW = g_fsLastFSWindow.lock()) {
                    const auto CUR = Compat::currentWindowBox(OLDW);

                    g_fsAssertBox    = CBox{CUR.x, CUR.y, kSpawnWidth,
                                            kSpawnHeight};
                    g_fsAssertFrames = 1;
                    g_fsStableCount  = 0;

                    Compat::setWindowBox(OLDW, g_fsAssertBox);
                }
            }
        }

        g_fsLastFSWindow = FSW;
    } else if (g_fsPhase == EFullscreenPhase::In2D) {
        // Exit passthrough when OUR window is no longer the fullscreen one
        // (toggled off, or fullscreen moved to another window entirely).
        const auto W = g_fsWindow.lock();

        if (!FSW || (W && FSW != W))
            startTo3D();
        else
            g_fsWasOn = true;
    } else if (g_fsPhase == EFullscreenPhase::To2D) {
        // Fullscreen was toggled off MID-ANIMATION: hand back to 3D from the
        // current animated pose instead of finishing onto a stale screen.
        if (!FSW)
            startTo3D();
    } else if (g_fsPhase == EFullscreenPhase::To3D) {
        g_fsWasOn = FSW != nullptr; // remembered for after the animation
    }
}

// To2D/To3D: pose the fullscreening window between its room pose and the
// screen-covering pose (smoothstep). Runs AFTER syncWorld, overriding the
// per-frame reseed.
static void applyFullscreenAnimation() {
    if (g_fsPhase != EFullscreenPhase::To2D && g_fsPhase != EFullscreenPhase::To3D)
        return;

    const auto W = g_fsWindow.lock();
    auto* E = W ? g_world.find(Compat::windowId(W)) : nullptr;

    if (!E) {
        // the window closed mid-transition: land the phase
        g_fsPhase  = EFullscreenPhase::None;
        g_fsWindow = {};
        return;
    }

    const float RAWP = fsRawProgress();
    g_fsRawP = RAWP; // feeds the composite alpha below
    float p = RAWP * RAWP * (3.0f - 2.0f * RAWP); // smoothstep

    // To2D: the SCREEN pose tracks the LIVE camera -- looking around during
    // the transition keeps the quad converging onto the current view, so the
    // handoff never pops.
    if (g_fsPhase == EFullscreenPhase::To2D) {
        const auto& CAM = g_scene.camera();
        const Vec3 FWD = CAM.forward();

        g_fsEndCenter = CAM.position + FWD * g_fsDistance;
        g_fsEndYaw    = -CAM.yaw;
        g_fsEndPitch  = CAM.pitch;
    }

    // To2D: room -> screen; To3D: screen -> room
    const auto  A = g_fsPhase == EFullscreenPhase::To2D;
    const Vec3  C0 = A ? g_fsStartCenter : g_fsEndCenter;
    const Vec3  C1 = A ? g_fsEndCenter : g_fsStartCenter;
    const float Y0 = A ? g_fsStartYaw : g_fsEndYaw;
    const float Y1 = A ? g_fsEndYaw : g_fsStartYaw;
    const float P0 = A ? g_fsStartPitch : g_fsEndPitch;
    const float P1 = A ? g_fsEndPitch : g_fsStartPitch;

    E->center = C0 + (C1 - C0) * p;
    E->yaw    = Y0 + wrapPi(Y1 - Y0) * p;
    E->pitch  = P0 + (P1 - P0) * p;

    // The roll fades out on the way to 2D (the compositor renders windows
    // unrolled) and fades back to the saved value on the way to the room.
    if (A)
        E->roll = g_fsSavedRoll * (1.0f - p);
    else
        E->roll = g_fsRollAtStart + (g_fsSavedRoll - g_fsRollAtStart) * p;

    // The REAL box animates between the floating box and the fullscreen box.
    // The client's buffer is stretched to this box by the compositor, so the
    // content scale inside the quad stays constant through the whole
    // transition. On the way back the box lands at the SPAWN size (960x540):
    // the post-FS size is deterministic and never inherits garbage from
    // earlier broken cycles.
    const CBox B0 = A ? g_fsRestoreBox : g_fsMonitorBox;
    const CBox B1 = A ? g_fsMonitorBox : g_fsRestoreBox;

    const CBox BOX{
        B0.x + (B1.x - B0.x) * p,
        B0.y + (B1.y - B0.y) * p,
        B0.w + (B1.w - B0.w) * p,
        B0.h + (B1.h - B0.h) * p,
    };

    if (auto W = g_fsWindow.lock())
        Compat::setWindowBox(W, BOX);

    // The quad follows the animated box at the room's pixel density. The
    // fullscreen quad maps 1:1 onto the monitor by definition, so the room's
    // window scale is suspended for its duration (syncWorld re-applies it
    // once the window lands back in the room). The scale itself is lerped
    // across the animation so the endpoints match their neighbours: the room
    // side is the window at config scale, the monitor side is exact 1:1 --
    // otherwise the quad visibly jumps at the handoff frames.
    E->spawnScale = 1.0f;

    const float ROOM_SCALE = configWindowScale();
    const float S0 = A ? ROOM_SCALE : 1.0f;
    const float S1 = A ? 1.0f : ROOM_SCALE;
    const float S = S0 + (S1 - S0) * p; // p is the eased progress

    E->width  = World3D::toWorld(BOX.w) * S;
    E->height = World3D::toWorld(BOX.h) * S;

    // The roll decays to zero: the 2D fullscreen view has no roll, and the
    // quad must land matching what the compositor will render.

    // Composite alpha: minimal fades at the handoffs (a couple of frames).
    // The snapshot and the live 2D render can be a frame apart, and a long
    // fade makes that desync visible; a 2-frame crossfade hides it.
    g_fsAlpha = A ?
        std::clamp(1.0f - (g_fsRawP - 0.96f) / 0.04f, 0.0f, 1.0f) :
        std::clamp(g_fsRawP / 0.04f, 0.0f, 1.0f);

    // Phase completion:
    //   To2D done -> hand the screen to the 2D compositor (In2D): input is
    //   released, the cursor reappears, the 3D scene stops rendering (the
    //   onRenderStage gate) and the pump polls for the fullscreen exit.
    //   To3D done -> restore the floating box (the client re-renders at its
    //   room size) and return to the plain 3D room.
    if (p >= 1.0f) {
        if (g_fsPhase == EFullscreenPhase::To2D) {
            g_fsPhase = EFullscreenPhase::In2D;

            Compat::setPointerCapture(false);

            // Our transition's per-frame setWindowBox calls polluted
            // Hyprland's remembered floating size with intermediate (up to
            // monitor-sized) values. Re-pin the captured PRE-FS size.
            if (auto W2 = g_fsWindow.lock())
                W2->m_target->rememberFloatingSize(
                    {g_fsRestoreBox.w, g_fsRestoreBox.h});

            if (Compat::setCursorHidden(false) && g_pHyprRenderer)
                g_pHyprRenderer->setCursorFromName("default", true);
        } else {
            // Rapid exit+reenter: the FS state may already be ON again for
            // this window -- fly it back to the screen (real 2D) instead of
            // dropping it into the room small while fullscreened.
            const auto MON = targetMonitor();
            const auto W2 = g_fsWindow.lock();

            if (W2 && MON &&
                Fullscreen::controller()->getFullscreenWindow(MON) == W2) {
                g_fsPhaseStart = std::chrono::steady_clock::now();
                startTo2D(W2, /*captureRestoreBox*/ false);
            } else {
                if (W2) {
                    // Home first: a window brought in front from another
                    // monitor goes back to the workspace it came from, so the
                    // box below lands where it lived.
                    if (g_fsHomeWindow.lock() == W2) {
                        if (const auto HOME = g_fsHomeWorkspace.lock())
                            Config::Actions::moveToWorkspace(HOME, true, W2);
                        g_fsHomeWindow    = {};
                        g_fsHomeWorkspace = {};
                    }

                    // The original position AND the pre-fullscreen size:
                    // Hyprland remembered the size when the FS was engaged,
                    // the box lerp above already animated the real box there.
                    Compat::setWindowBox(W2, g_fsRestoreBox);

                    // Hyprland's own fullscreen-exit restore animates the
                    // window toward ITS remembered floating size -- which our
                    // transition kept updating -- and can retarget the box
                    // AFTER this point. Re-assert the restore box for a few
                    // frames so the final size is ours.
                    // The SAME box the quad just landed at (the spawn
                    // size) -- asserting the raw g_fsRestoreBox here made
                    // our own guard resize the window back to the polluted
                    // monitor size right after the landing.
                    g_fsAssertBox    = g_fsRestoreBox;
                    g_fsAssertFrames = 1;    // condition-driven: runs until stable
                    g_fsStableCount  = 0;

                    // The FS fade dimmed EVERY other workspace window to
                    // alpha 0 when the fullscreen started; our transition can
                    // leave the OUT fade mid-flight. Reset the channel
                    // explicitly, or windows stay barely visible.
                    if (MON && MON->m_activeWorkspace) {
                        for (auto const& W3 :
                             Desktop::windowState()->windows()) {
                            if (W3 && W3->m_workspace ==
                                          MON->m_activeWorkspace &&
                                !W3->m_pinned)
                                *W3->alpha(
                                     Desktop::View::WINDOW_ALPHA_FULLSCREEN) =
                                    1.F;
                        }
                    }
                }

                g_fsPhase = EFullscreenPhase::None;
            }
        }
    }
}

// Ends a model roll gesture: restores the suspended gravity on every exit
// path (button release, crosshair leaving, model vanishing) and WAKES the
// body -- during the roll it was held motionless, so Jolt put it to sleep,
// and a sleeping body would stay frozen forever even with gravity back.
static void endModelRoll() {
    if (s_wheelRot.model && s_wheelRot.modelIndex < g_joltBodies.size()) {
        const auto BODY = g_joltBodies[s_wheelRot.modelIndex].body;
        g_bodyIf->SetGravityFactor(BODY, g_rolledGravity);
        g_bodyIf->ActivateBody(BODY);
    }
    s_wheelRot.model      = false;
    s_wheelRot.active     = false;
    s_wheelRot.modelIndex = SIZE_MAX;
}

// Per-frame roll gesture: rotate the window around its normal by the signed
// angle the crosshair swept around the window center (measured in world
// space around the normal -- invariant to the roll itself, so the gesture
// never feeds back into its own measurement).
static void updateWheelRoll() {
    if (!s_wheelRot.active)
        return;

    if (s_wheelRot.model) {
        // A config reload can shrink the object list mid-roll.
        if (s_wheelRot.modelIndex >= g_sceneObjects.size()) {
            endModelRoll();
            return;
        }

        auto* MODEL = g_scene.sceneModel(s_wheelRot.modelIndex);
        if (!MODEL || !MODEL->loaded()) {
            endModelRoll();
            return;
        }

        const auto& CAM = g_scene.camera();
        Vec3 P;
        if (!rayPlanePoint(CAM.position, CAM.centerRay(), s_wheelRot.center,
                           s_wheelRot.normal, P))
            return;

        const Vec3 V = P - s_wheelRot.center;
        const float ANGLE = std::atan2(
            dot(cross(s_wheelRot.reference, V), s_wheelRot.normal),
            dot(s_wheelRot.reference, V));

        constexpr float DEG = 3.14159265358979f / 180.0f;
        auto& ROT = g_sceneObjects[s_wheelRot.modelIndex].rotationDeg;
        ROT.z = (s_wheelRot.startRoll + ANGLE) / DEG;

        g_scene.setSceneObjectTransform(s_wheelRot.modelIndex,
                                        MODEL->position(), ROT);

        // Rolling a physics object suspends its simulation: gravity off,
        // velocities zeroed -- otherwise the falling/dragging motion fights
        // the user's rotation and the behavior looks erratic. The new
        // rotation is ALSO pushed into the body: the per-frame readback
        // reads the body transform, and without this it would revert the
        // roll every frame.
        if (s_wheelRot.model && s_wheelRot.modelIndex < g_joltBodies.size()) {
            const auto JB = g_joltBodies[s_wheelRot.modelIndex].body;
            g_bodyIf->SetGravityFactor(JB, 0.f);
            g_bodyIf->SetLinearVelocity(JB, JPH::Vec3::sZero());
            g_bodyIf->SetAngularVelocity(JB, JPH::Vec3::sZero());
            g_bodyIf->SetPositionAndRotationWhenChanged(
                JB,
                JPH::RVec3(MODEL->position().x, MODEL->position().y,
                           MODEL->position().z),
                eulerToQuat(ROT), JPH::EActivation::DontActivate);
        }

        damageCurrentMonitor();
        return;
    }

    const auto HIT = aimHit();

    if (!HIT.hit || HIT.id != s_wheelRot.id) {
        s_wheelRot.active = false; // crosshair left the window: end it
        return;
    }

    auto* ENTITY = g_world.find(s_wheelRot.id);

    if (!ENTITY) {
        s_wheelRot.active = false;
        return;
    }

    const Vec3 V = HIT.point - s_wheelRot.center;
    const float ANGLE = std::atan2(
        dot(cross(s_wheelRot.reference, V), s_wheelRot.normal),
        dot(s_wheelRot.reference, V));

    // Wrapped to [-pi, pi]: the angle is 2pi-periodic, so the visual is
    // identical, but the fullscreen transition interpolates this value
    // linearly -- an unwrapped 700-degree roll would unwind like a top.
    ENTITY->roll = wrapPi(s_wheelRot.startRoll + ANGLE);
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

    // A finished resize gesture: schedule one forced snapshot of the resized
    // window for the next capture pass -- its purpose is the silhouette's
    // settled full-resolution refresh (see CWindowCapture::refreshSkirtMask).
    if (g_resize.active)
        g_skirtFinalRefreshId = g_resize.id;

    g_resize = {};
    // A finished roll gesture restores the object's gravity.
    endModelRoll();

    // Dropping a carried scene object: it re-enters every contact pair at
    // its current position (and its BVH tree rebuilds for picking).
    if (g_bodyIf && g_mapGrabIndex != SIZE_MAX &&
        g_mapGrabIndex < g_joltBodies.size() &&
        g_joltBodies[g_mapGrabIndex].valid)
        g_bodyIf->SetObjectLayer(
            g_joltBodies[g_mapGrabIndex].body,
            g_sceneObjects[g_mapGrabIndex].dynamic ? LAYER_MOVING
                                                    : LAYER_STATIC);
    g_mapGrabIndex = SIZE_MAX;
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

// F8: use a window as one uses it at a desk -- the TV's from the couch, or
// any other. The camera holds still, the mouse moves a pointer across the
// window (the crosshair drawn where it is), buttons and wheel act there, and
// the window keeps the keyboard. F8 again gives the room its mouse back.
// Escape stays the window's: a browser or a game needs it. Super+F still
// takes the window to the real fullscreen and back.
struct SUse {
    std::uintptr_t id = 0;
    Vector2D       local{};     // the pointer, surface-local px
    EKeyboardMode  mode{};      // to restore
    std::uintptr_t lockBefore = 0;
};
static SUse g_use;

static void useEnd() {
    if (!g_use.id)
        return;
    g_keyboardMode = g_use.mode;
    g_focusLockId  = g_use.lockBefore;
    g_use = {};
    g_scene.setCursorPoint({}, false);
    damageCurrentMonitor();
}

static void useBegin() {
    const auto HIT    = aimHit();
    const auto TARGET = HIT.hit ? targetFromHit(HIT.id) : SHitTarget{};
    if (!TARGET.window) {
        notify("[hypr3d] F8: aim at a window to use it", CHyprColor{0.2f, 0.8f, 0.4f, 1.0f});
        return;
    }
    g_use.id         = Compat::windowId(TARGET.window);
    g_use.local      = localFromHit(HIT);
    g_use.mode       = g_keyboardMode;
    g_use.lockBefore = g_focusLockId;
    if (Compat::focusedWindow() != TARGET.window)
        Compat::focusWindow(TARGET.window);
    g_focusLockId  = g_use.id;
    g_keyboardMode = EKeyboardMode::Window;
    resetMovementKeys();
    damageCurrentMonitor();
}

// Every frame: where the pointer is in the world, for the crosshair.
static void useTick() {
    if (!g_use.id)
        return;
    const auto* E = g_world.find(g_use.id);
    if (!E || !Compat::findWindowById(g_use.id) || E->logicalWidth <= 0 || E->logicalHeight <= 0) {
        useEnd(); // it closed
        return;
    }
    const float U = (static_cast<float>(g_use.local.x) + E->surfaceOffsetX) / E->logicalWidth;
    const float V = (static_cast<float>(g_use.local.y) + E->surfaceOffsetY) / E->logicalHeight;
    g_scene.setCursorPoint(E->center + g_world.rightOf(g_use.id) * ((U - 0.5f) * E->width) +
                               g_world.upOf(g_use.id) * ((0.5f - V) * E->height),
                           true);
}

// The mouse moving the pointer of the window in use.
static void useMove(double dx, double dy) {
    const auto  W = Compat::findWindowById(g_use.id);
    const auto* E = g_world.find(g_use.id);
    if (!W || !E) {
        useEnd();
        return;
    }
    g_use.local.x = std::clamp(g_use.local.x + dx, 0.0, static_cast<double>(E->surfaceWidth));
    g_use.local.y = std::clamp(g_use.local.y + dy, 0.0, static_cast<double>(E->surfaceHeight));
    Compat::deliverMotion(W, g_use.local, inputTimeMs());
    damageCurrentMonitor();
}

static void forwardPointerToAim(uint32_t timeMs) {
    if (!ownsInput() || g_pointerDown || g_use.id)
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

// A game that locked the pointer (zwp_locked_pointer, as CS2 and Minecraft
// do once they take the mouse) has the mouse and the keyboard: Hyprland sends
// it the relative motion itself, the room's camera holds still and the keys
// reach it -- until it lets go (its own Escape), as the cursor holds still
// for it in 2D. Turning the room as well moved the window under the crosshair,
// and the game read the positions the room forwarded as more motion: 100 for
// 40 under Xwayland, jumps of 1000 px for a game that warps (measured
// 2026-10-07).
static bool gameHasMouse() {
    const auto W = Compat::focusedWindow();
    return W && g_world.find(Compat::windowId(W)) && Compat::pointerConstraintOf(W) == 2;
}

// Absolute-position fallback for installations where the PointerManager hook
// is unavailable. When the hook works, it feeds onPointerMotion directly and
// this event only keeps the baseline warm.
static bool onPointerMotion(double dx, double dy) {
    if (!ownsInput())
        return false;
    if (gameHasMouse())
        return true; // the game had the motion already; the cursor stays put
    if (g_use.id) {
        useMove(dx, dy);
        return true;
    }

    g_input.addMotion(dx, dy);
    damageCurrentMonitor();
    return true;
}

static void onRenderPre(PHLMONITOR mon) {
    if (g_capturing)
        return;

    const auto TARGET = targetMonitor();
    if (TARGET && mon != TARGET) {
        // A neighbour the room spans draws it as well (onRenderStage); the
        // target's frame captures and lays out for all of them.
        g_currentRenderMon = g_active && isSpanned(mon) ? mon : nullptr;
        return;
    }

    g_currentRenderMon = mon;

    if (!g_active) {
        g_monitor = mon;
        return;
    }

    g_monitor = mon;
    serviceCapture();
}


// --- lifecycle --------------------------------------------------------------

// Takes the live room into g_room. Surfaces not in the room right now (on
// another workspace, or not captured yet) keep what was remembered of them;
// closed ones are dropped.
static void rememberRoom() {
    const auto& CAM = g_scene.camera();
    g_room.valid    = true;
    g_room.eye      = CAM.position;
    g_room.yaw      = CAM.yaw;
    g_room.pitch    = CAM.pitch;
    g_room.viewMode = g_viewMode;

    std::erase_if(g_room.poses, [](const auto& KV) {
        return !KV.second.window.lock() && !KV.second.layer.lock();
    });

    for (const auto& E : g_world.entities()) {
        SRememberedPose P;
        if (const auto W = Compat::findWindowById(E.id))
            P.window = W;
        else if (const auto L = Compat::findLayerById(E.id))
            P.layer = L;
        else
            continue;
        P.center = E.center;
        P.yaw    = E.yaw;
        P.pitch  = E.pitch;
        P.roll   = E.roll;
        g_room.poses[E.id] = P;
    }
}

static void deactivate3D() {
    endReading(true);
    // Windows still in the bin, their applications not gone yet: back where
    // they were let go, not remembered tiny.
    for (const auto& T : g_trashed)
        if (auto* E = g_world.find(T.id)) {
            E->center = T.fromCenter;
            E->yaw = T.fromYaw, E->pitch = T.fromPitch, E->roll = T.fromRoll;
        }
    g_trashed.clear();
    rememberRoom();

    finishClientButton(0);
    resetPointerGesture();

    g_active = false;
    syncRoomState(true);
    abortSight("The room closed before the picture was taken");
    stopFramePump();

    // Fullscreen passthrough state: back to plain 3D-off. Restore the real
    // box if a transition was mid-flight (the window would otherwise stay
    // monitor-sized). The SIZE is the spawn size -- never the possibly
    // polluted restore box.
    if (auto W = g_fsWindow.lock())
        Compat::setWindowBox(
            W,
            CBox{g_fsRestoreBox.x, g_fsRestoreBox.y, kSpawnWidth,
                 kSpawnHeight});

    g_fsPhase        = EFullscreenPhase::None;
    g_fsWindow       = {};
    g_fsWasOn        = false;
    g_fsAlpha        = 1.0f;
    g_fsFade         = 1.0f;
    g_fsAssertFrames = 0;

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
    damageAllMonitors();
    g_renderedOnce = false;
}

// The close transition finishes inside render.stage, i.e. between the frame's
// startRenderPass() and endRender(). deactivate3D destroys the snapshot
// framebuffers and restores the layout, which broke the compositor's outer
// endRender() (SEGV in CMonitor::useFP16) -- the same failure syncWorld
// documents. Run it from the event loop instead; PLUGIN_EXIT cancels a pending
// call so it never runs into unloaded code.
static uint64_t g_deactivateLater = 0;

static void requestDeactivate3D() {
    if (g_deactivateLater || !g_pEventLoopManager)
        return;

    g_deactivateLater = g_pEventLoopManager->doLater([] {
        g_deactivateLater = 0;

        // Reopened before the deferred teardown ran.
        if (!g_active || g_transitionTarget > 0.0f)
            return;

        deactivate3D();
        damageCurrentMonitor();
    });
}

// Player spawn point (config player_spawn): the coordinates are the player's
// FEET, so spawning at 0,0,0 stands on the grid platform at world zero instead
// of falling through it. Eyes ride kEyeHeight above.
static void placeAtSpawn() {
    auto& CAM = g_scene.camera();
    CAM.position = Vec3{
        g_playerSpawn.x,
        g_playerSpawn.y + Camera::kEyeHeight,
        g_playerSpawn.z,
    };

    // Camera forward is {sin yaw, ., -cos yaw}: looking at the origin
    // from (x, z) means yaw = atan2(-x, z).
    CAM.yaw = std::atan2(-g_playerSpawn.x, g_playerSpawn.z);
    // Walking, the eye looks level: a wall of monitors built now stands
    // upright on the floor instead of leaning back.
    if (!g_playerFlying)
        CAM.pitch = 0.0f;
}

static void enter3D() {
    // Entered again before the room closed (toggled while it faded out, or
    // opened while open): what is live now is what to come back to.
    if (g_active)
        rememberRoom();

    // The eye's monitor for this visit: where the room was opened from. The
    // fullscreen windows already on the other monitors stay where they are.
    if (!g_active) {
        g_viewMonitor      = targetMonitor();
        g_fsElsewhereKnown = false;
        g_fsHomeWindow     = {};
        g_fsHomeWorkspace  = {};
    }

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

    // Back where the player stood and looked when the room closed; the spawn
    // point the first time and after hl.plugin.hypr3d.reset().
    if (g_room.valid) {
        auto& CAM    = g_scene.camera();
        CAM.position = g_room.eye;
        CAM.yaw      = g_room.yaw;
        CAM.pitch    = g_room.pitch;
    } else
        placeAtSpawn();

    g_capture.releaseAll();

    resetMovementKeys();

    g_grounded = false;
    g_viewMode = g_room.valid ? g_room.viewMode : 0;
    g_scene.camera().mirrorView = false;
    g_scene.setPlayerVisible(false);
    g_scene.setPlayerDebugCapsule(Vec3{}, false);

    g_keyboardMode = EKeyboardMode::Space;
    g_altHeld      = false;
    s_zoomId       = 0;

    // A fullscreen window that already exists: it joins the room as a
    // standard spawn-sized floating panel (a fullscreened/tiled box would
    // otherwise enter the room monitor-sized AND pollute the stable-box
    // memory). The passthrough still triggers on the fullscreen EVENT.
    if (const auto MON = targetMonitor()) {
        if (const auto FSW =
                Fullscreen::controller()->getFullscreenWindow(MON)) {
            Compat::setWindowBox(
                FSW,
                CBox{MON->m_position.x + MON->m_size.x * 0.5 -
                         kSpawnWidth * 0.5,
                     MON->m_position.y + MON->m_size.y * 0.5 -
                         kSpawnHeight * 0.5,
                     kSpawnWidth, kSpawnHeight});

            g_fsLastFSWindow = FSW;
        }

        g_fsWasOn =
            Fullscreen::controller()->getFullscreenWindow(MON) != nullptr;
    } else
        g_fsWasOn = false;

    g_fsStableBoxes.clear();

    g_renderedOnce = false;
    g_captureFrames = 0;
    g_reportedFramebufferError = false;
    g_reportedRenderError = false;

    // The first full live capture is performed from render.pre, where
    // Hyprland has not entered its main render pass yet.
}

static void toggle3D() {
    if (sessionLocked() && g_transitionTarget <= 0.5f)
        return;
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
    if (sessionLocked())
        return;
    g_transitionTarget = 1.0f;
    enter3D();
    damageCurrentMonitor();

    notify(
        "[hypr3d] open",
        CHyprColor{0.2f, 0.8f, 0.4f, 1.0f}
    );
}

static void useEnd();

static void close3D() {
    useEnd();
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

// World-space glide velocity for the WASD inertia (units/second). Lives
// across frames so releasing the keys coasts down instead of cutting dead.
static Vec3 s_moveVel{};

// Walking is a game, not a camera on rails (the owner, 2026-10-07: "fühlt
// sich sehr stiff an"): Quake's ground model -- friction, then acceleration
// toward the wished speed -- and little steering in the air, so a jump
// keeps its momentum. Coyote time and a jump buffer forgive Space pressed a
// moment late or early; letting go of Space while rising cuts the jump.
// Two presses of Space toggle flight, as in Minecraft; Shift crouches when
// walking and sinks when flying; Ctrl sprints and widens the view a little.
static constexpr float kGroundAccel    = 10.0f; // Quake/Source sv_accelerate
static constexpr float kGroundFriction = 6.0f;  // sv_friction
static constexpr float kStopSpeed      = 1.5f;  // m/s, friction's floor (sv_stopspeed)
static constexpr float kAirAccel       = 2.0f;  // steering in the air
static constexpr float kJumpSpeed      = 5.5f;  // m/s up: ~1.1 m at 14 m/s^2
static constexpr float kCoyoteTime     = 0.12f; // s after leaving an edge
static constexpr float kJumpBuffer     = 0.12f; // s a press waits for the ground
static constexpr float kDoubleTap      = 0.30f; // s between the two presses
static constexpr float kCrouchSpeed    = 0.45f; // of the walking speed
static constexpr float kCrouchDrop     = 0.40f; // m the eye sinks
// Crouch, landing dip and bob together stay under 0.45 m: further apart,
// camera and body read as a teleport (update3D, 0.5 m) and the body would
// be moved into the floor.
static constexpr float kViewOffsetMax  = 0.45f;

static double s_lastGroundedAt   = -1.0;
static double s_jumpPressedAt    = -1.0;
static double s_lastSpacePressAt = -1.0;
static bool   s_jumpRising       = false;
static bool   s_wasGrounded      = false;
static float  s_lastFallSpeed    = 0.0f;
static float  s_viewDrop         = 0.0f; // crouch, m below the eye
static float  s_landDip          = 0.0f; // landing, m below the eye

static double nowSeconds() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Quake's PM_Accelerate: add speed along the wished direction up to the
// wished speed, at most accel * wishspeed per second.
static void accelerateToward(Vec3& vel, const Vec3& dir, float wish, float accel,
                             float dt) {
    const float CURRENT = vel.x * dir.x + vel.z * dir.z;
    const float ADD = wish - CURRENT;
    if (ADD <= 0.0f)
        return;
    const float STEP = std::min(accel * dt * wish, ADD);
    vel.x += dir.x * STEP;
    vel.z += dir.z * STEP;
}

// Head tracking (config head = { enabled, port, look, move }), off unless
// asked for: opentrack's "UDP over network" output -- six doubles, x y z in
// cm and yaw pitch roll in degrees -- on 127.0.0.1:port (4242, opentrack's
// default); the face tracking itself is opentrack's, from any webcam. look
// turns the view by the head's turn (degrees per degree), move shifts the
// eye by the head's shift (metres per metre): looking past a window's edge
// by leaning, as Compiz did in 2010 and Breezy Desktop does with opentrack.
// The Reddit finds praise it and also report nausea and jitter
// (research/hypr3d-ideen-2026-10-06.md), so: off by default, smoothed, a
// jump of more than 25 cm or 60 degrees in one packet thrown away, and
// back to the centre a second after the packets stop. First person only.
// Unverified with a real tracker: the signs (a tracker's yaw to the right is
// taken as looking right); look or move < 0 turns an axis round.
static bool  g_cfgHead     = false;
static float g_cfgHeadPort = 4242.0f;
static float g_cfgHeadLook = 1.0f;
static float g_cfgHeadMove = 1.0f;
struct SHead {
    int              fd   = -1;
    int              port = 0;
    wl_event_source* source = nullptr;
    double           target[6]{}; // the last packet: x y z cm, yaw pitch roll deg
    double           shown[6]{};  // smoothed
    bool             any = false;
    std::chrono::steady_clock::time_point last;
    uint64_t         packets = 0, dropped = 0;
};
static SHead g_head;
static float s_headYawApplied = 0.0f, s_headPitchApplied = 0.0f; // rad, in CAM

static int headReadable(int fd, uint32_t, void*) {
    double buf[8];
    for (;;) {
        const ssize_t N = recv(fd, buf, sizeof(buf), 0);
        if (N < 0)
            break; // EAGAIN: all read
        if (N != 6 * sizeof(double))
            continue;
        bool ok = true;
        for (int i = 0; i < 6; ++i)
            ok = ok && std::isfinite(buf[i]) && std::fabs(buf[i]) < 1000.0;
        if (!ok) {
            ++g_head.dropped;
            continue;
        }
        if (g_head.any) {
            const double MOVED = std::hypot(buf[0] - g_head.target[0], buf[1] - g_head.target[1], buf[2] - g_head.target[2]);
            const double TURNED = std::max(std::fabs(buf[3] - g_head.target[3]), std::fabs(buf[4] - g_head.target[4]));
            if (MOVED > 25.0 || TURNED > 60.0) {
                ++g_head.dropped;
                continue;
            }
        }
        std::copy(buf, buf + 6, g_head.target);
        g_head.any  = true;
        g_head.last = std::chrono::steady_clock::now();
        ++g_head.packets;
    }
    damageCurrentMonitor();
    return 0;
}

static void headClose() {
    if (g_head.source)
        wl_event_source_remove(g_head.source);
    if (g_head.fd >= 0)
        close(g_head.fd);
    g_head = {};
}

// Opens, moves or closes the socket after a config change.
static void headApply() {
    const int PORT = std::clamp(static_cast<int>(g_cfgHeadPort), 1, 65535);
    if (!g_cfgHead || PORT != g_head.port)
        headClose();
    if (!g_cfgHead || g_head.fd >= 0 || !g_pCompositor || !g_pCompositor->m_wlEventLoop)
        return;
    const int FD = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    sockaddr_in at{};
    at.sin_family      = AF_INET;
    at.sin_port        = htons(static_cast<uint16_t>(PORT));
    at.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // this machine only
    if (FD < 0 || bind(FD, reinterpret_cast<sockaddr*>(&at), sizeof(at)) != 0) {
        if (FD >= 0)
            close(FD);
        notify("[hypr3d] head tracking: 127.0.0.1:" + std::to_string(PORT) + " is taken or refused", CHyprColor{1.0f, 0.3f, 0.2f, 1.0f});
        return;
    }
    g_head.fd     = FD;
    g_head.port   = PORT;
    g_head.source = wl_event_loop_add_fd(g_pCompositor->m_wlEventLoop, FD, WL_EVENT_READABLE, headReadable, nullptr);
}

// The head's offsets for this frame, eased: towards the last packet, or back
// to the centre once the packets stopped.
static bool headBusy() {
    if (!g_cfgHead || !g_head.any)
        return false;
    for (int i = 0; i < 6; ++i)
        if (std::fabs(g_head.shown[i] - g_head.target[i]) > 0.01)
            return true;
    return std::chrono::duration<float>(std::chrono::steady_clock::now() - g_head.last).count() < 1.2f;
}

static void headStep(float dt) {
    if (g_cfgHead && g_head.any &&
        std::chrono::duration<float>(std::chrono::steady_clock::now() - g_head.last).count() > 1.0f)
        std::fill(g_head.target, g_head.target + 6, 0.0); // tracking stopped: back to the centre
    const double K = 1.0 - std::exp(-dt / 0.05);
    for (int i = 0; i < 6; ++i) {
        const double TO = g_cfgHead ? g_head.target[i] : 0.0;
        g_head.shown[i] += (TO - g_head.shown[i]) * K;
    }
}

// Walk bob (view-only): sine phase in radians + eased amplitude. The body
// pose stays physical -- the offset rides on the camera each frame and is
// re-derived from the clean eye point, so it never feeds back into Jolt.
static float s_bobPhase = 0.0f;
static float s_bobAmp   = 0.0f;
static constexpr float kBobAmplitude = 0.035f; // meters at full walk speed
static constexpr float kBobRate      = 6.0f;   // radians per meter walked
static constexpr float kBobRoll      = 0.008f; // radians (~0.9 deg) head tilt

// Computes the key-driven world-space velocity into s_moveVel (units/s)
// with the move-inertia glide. The PLAYER BODY applies it -- Jolt owns the
// pose, so this no longer moves the camera directly.
static void applyCameraMovement(float dt) {
    float FORWARD  = (g_keyFwd ? 1.f : 0.f) - (g_keyBack ? 1.f : 0.f);
    float STRAFE   = (g_keyRight ? 1.f : 0.f) - (g_keyLeft ? 1.f : 0.f);
    float VERTICAL = (g_keyUp ? 1.f : 0.f) - (g_keyDown ? 1.f : 0.f);

    auto& CAM = g_scene.camera();

    // Walking mode: Shift does nothing (gravity owns vertical), Space is a
    // jump impulse queued in the key handler.
    if (!g_playerFlying)
        VERTICAL = 0.f;

    // Diagonals must not be faster than a straight run: W+D used to add the
    // two key speeds into a sqrt(2)x diagonal.
    {
        const float LEN = std::sqrt(FORWARD * FORWARD + STRAFE * STRAFE);
        if (LEN > 1.0f) {
            FORWARD /= LEN;
            STRAFE /= LEN;
        }
    }

    const bool  MOVING = FORWARD != 0.f || STRAFE != 0.f || VERTICAL != 0.f;
    const float tau    = g_cfgMoveInertia;

    if (!g_playerFlying) {
        // --- walking: friction and acceleration on the ground, a little
        // steering in the air; Jolt's gravity owns the vertical axis ---
        const bool  CROUCH = g_keyDown;
        const float WISH = CAM.moveSpeed * (g_keySprint && !CROUCH ? 2.5f : 1.0f) *
            (CROUCH ? kCrouchSpeed : 1.0f);
        Vec3 dir = CAM.flatForward() * FORWARD + CAM.right() * STRAFE;
        const float DLEN = std::sqrt(dir.x * dir.x + dir.z * dir.z);
        if (DLEN > 1e-4f)
            dir = dir * (1.0f / DLEN);

        // From what the body really does: a wall or a prop takes its share.
        Vec3 vel{s_moveVel.x, 0.f, s_moveVel.z};
        if (!g_playerBody.IsInvalid() && g_bodyIf) {
            const auto BV = g_bodyIf->GetLinearVelocity(g_playerBody);
            vel = Vec3{BV.GetX(), 0.f, BV.GetZ()};
        }
        if (g_grounded) {
            const float SPEED = std::sqrt(vel.x * vel.x + vel.z * vel.z);
            if (SPEED > 0.0f) {
                const float DROP = std::max(SPEED, kStopSpeed) * kGroundFriction * dt;
                const float SCALE = std::max(0.0f, SPEED - DROP) / SPEED;
                vel.x *= SCALE;
                vel.z *= SCALE;
            }
            if (DLEN > 1e-4f)
                accelerateToward(vel, dir, WISH, kGroundAccel, dt);
        } else if (DLEN > 1e-4f)
            accelerateToward(vel, dir, WISH, kAirAccel, dt);
        s_moveVel = Vec3{vel.x, 0.f, vel.z};
        (void)tau;
        return;
    }

    // --- flying: all three axes are key-driven ---
    const float SPEED =
        CAM.moveSpeed * (g_keySprint ? 2.5f : 1.0f) * (MOVING ? 1.0f : 0.0f);
    const Vec3 TARGET =
        CAM.flatForward() * (FORWARD * SPEED) +
        CAM.right() * (STRAFE * SPEED) +
        Vec3{0.f, 1.f, 0.f} * (VERTICAL * SPEED);

    if (tau <= 0.0f || dt <= 0.0f) {
        s_moveVel = TARGET;
        return;
    }

    // Glide toward the key-driven velocity with tau as the time constant;
    // with the keys released the target is zero, so motion decays smoothly.
    s_moveVel += (TARGET - s_moveVel) * (1.0f - std::exp(-dt / tau));

    // Snap the decay tail off once it is far below a frame of movement, so
    // the glide always terminates.
    if (!MOVING &&
        std::fabs(s_moveVel.x) + std::fabs(s_moveVel.y) +
            std::fabs(s_moveVel.z) < 0.01f)
        s_moveVel = {};
}

static void update3D(float dt) {
    // Release capture unconditionally when the view no longer owns input, so
    // a disappearing monitor or a closing transition can never strand the
    // pointer in captured mode.
    Compat::setPointerCapture(g_hookInstalled && ownsInput());

    float yawDelta = 0.0f;
    float pitchDelta = 0.0f;

    // Look/movement stay live during the fullscreen transition: the To2D end
    // pose tracks the camera, so the quad keeps converging onto the view.
    // Plugin settings come from lua (hl.plugin.hypr3d.config) and are
    // already clamped at set time.
    g_input.setLookSmoothing(g_cfgLookInertia);
    g_scene.camera().moveSpeed = g_cfgMoveSpeed;
    g_input.setSensitivity(g_cfgSensitivity);

    // Map: the scene owns the file/GL side; collision rebuilds its BVH
    // whenever the map (re)loaded.
    std::vector<GLScene::SSceneSpec> SPECS;
    SPECS.reserve(g_sceneObjects.size());

    for (const auto& OBJ : g_sceneObjects) {
        GLScene::SSceneSpec SPEC;
        SPEC.path          = OBJ.path;
        SPEC.position      = OBJ.position;
        SPEC.rotationDeg   = OBJ.rotationDeg;
        SPEC.scale         = OBJ.scale;
        SPEC.emissiveScale = OBJ.emissiveScale;
        SPEC.flat          = OBJ.flat;
        SPEC.center        = OBJ.center;
        SPEC.centerOffset  = OBJ.centerOffset;
        SPECS.push_back(SPEC);
    }

    g_scene.setSceneObjects(SPECS);
    g_scene.setGridVisible(g_cfgGrid);

    // Frame rate for the F3 HUD: slow exponential average over dt.
    if (dt > 0.0f)
        g_debugFps = g_debugFps * 0.9f + (1.0f / dt) * 0.1f;
    g_scene.setDebugFps(g_debugFps);

    // Window alpha keepalive lives on the FS pump (damageWindowsWithLiveAlpha)
    // -- it must tick even when a frame is slow.

    // Collision = PER-OBJECT trees: each scene object owns a small BVH over
    // its world triangles, rebuilt only when ITS OWN generation changes.
    // The static map's big tree builds once and never rebuilds for the FS
    // cycles or object carries -- the release hitch was the full 70k-triangle
    // rebuild; now a release rebuilds only the carried prop's small tree.
    // The carried object's tree also rebuilds per frame (a prop is small)
    // but is EXCLUDED from the queries while carried, so the object the
    // player holds cannot push them around.
    // --- Jolt: rigid body simulation for scene objects -------------------
    // Bodies live in Jolt; transforms read back into the config objects (the
    // render + collision layers follow the config as always). Jolt sleeps
    // resting bodies, so a settled scene costs nothing.
    const auto JOLT_T0 = std::chrono::steady_clock::now();
    joltSyncBodies();

    // While a scene object is still being decoded, or its first shape
    // cooked, its collision is missing: the player and every prop would fall
    // through a map that is not there yet. Time stands still until it is --
    // as it did while the load froze the whole compositor.
    if (g_joltSystem && !g_scene.scenePending() && !joltShapesPending()) {
        // All jobs execute on the main thread (0 workers), but through the
        // FULL thread-pool job system: it handles the dependency graph of
        // PhysicsSystem::Update, unlike JobSystemSingleThreaded, which
        // asserts (SIGTRAP) as soon as a dynamic body creates dependent
        // integration jobs.
        static JPH::TempAllocatorMalloc TEMP_ALLOC;
        static JPH::JobSystemThreadPool JOB_SYSTEM(
            JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, 0);
        const float SIM_DT = std::clamp(dt, 1.0f / 240.0f, 1.0f / 30.0f);
        // Sub-step the sim. One step of up to 1/30 moves a 5 m/s body 16 cm
        // -- enough to hop onto seam/trim geometry and fall back every few
        // frames (the wall-press jitter; 8.8 cm of position bounce measured
        // against the real map, zero with sub-steps). 3 sub-steps cap the
        // effective step at 1/90.
        g_joltSystem->Update(SIM_DT, 3, &TEMP_ALLOC, &JOB_SYSTEM);

        for (size_t i = 0; i < g_sceneObjects.size(); ++i) {
            auto& JB = g_joltBodies[i];
            auto& OBJ = g_sceneObjects[i];

            if (!OBJ.dynamic || !OBJ.physics || !JB.valid || i == g_mapGrabIndex)
                continue;

            const auto POS = g_bodyIf->GetPosition(JB.body);
            const auto ROT = g_bodyIf->GetRotation(JB.body);

            const Vec3 NEW_POS{POS.GetX(), POS.GetY(), POS.GetZ()};
            const Vec3 NEW_ROT = quatToEuler(ROT);

            const bool CHANGED =
                std::fabs(NEW_POS.x - OBJ.position.x) > 1e-6f ||
                std::fabs(NEW_POS.y - OBJ.position.y) > 1e-6f ||
                std::fabs(NEW_POS.z - OBJ.position.z) > 1e-6f ||
                std::fabs(NEW_ROT.x - OBJ.rotationDeg.x) > 1e-4f ||
                std::fabs(NEW_ROT.y - OBJ.rotationDeg.y) > 1e-4f ||
                std::fabs(NEW_ROT.z - OBJ.rotationDeg.z) > 1e-4f;

            if (CHANGED) {
                OBJ.position    = NEW_POS;
                OBJ.rotationDeg = NEW_ROT;
                g_scene.setSceneObjectTransform(i, NEW_POS, NEW_ROT);
            }
        }

        // Player readback: the camera derives from the body's eye point --
        // first person, or third person behind/front (F5) along the look
        // axes (the front view looks back at the character). While a
        // fullscreen transition animates the transition owns the camera.
        // (g_transition is the ROOM progress -- 1.0 in normal 3D -- it must
        // NOT gate this.)
        if (!g_playerBody.IsInvalid()) {
            const auto PPOS = g_bodyIf->GetPosition(g_playerBody);
            const Vec3 EYE{PPOS.GetX(), PPOS.GetY() + PLAYER_EYE_OFF,
                           PPOS.GetZ()};
            auto& CAM = g_scene.camera();

            if (g_fsPhase == EFullscreenPhase::None && g_viewMode == 0) {
                // First person: the camera IS the eye. The third-person
                // orbit is applied AFTER this frame's look update (see the
                // movement block) -- computing it here would position the
                // camera by the OLD yaw while the view points by the NEW
                // one, and the whole world would step on every turn.
                if (EYE.x != CAM.position.x || EYE.y != CAM.position.y ||
                    EYE.z != CAM.position.z) {
                    CAM.position = EYE;
                    damageCurrentMonitor();
                }
            }
            CAM.mirrorView = g_viewMode == 2;

            g_grounded = playerGrounded();

            // The character renders in third person only; F3 adds the
            // capsule outline.
            g_scene.setPlayerVisible(g_viewMode != 0);
            g_scene.setPlayerDebugCapsule(
                Vec3{PPOS.GetX(), PPOS.GetY(), PPOS.GetZ()}, g_debugHud);

            // Character state: air = jump (plays once and holds), grounded =
            // walk/run by horizontal speed, else idle. The POSE (the facing)
            // is applied after this frame's look update below -- applying it
            // here would render the model one camera-frame behind, stepping
            // after the mouse on every turn.
            if (g_scene.player()->loaded()) {
                const auto VEL = g_bodyIf->GetLinearVelocity(g_playerBody);
                const float HSP = std::sqrt(VEL.GetX() * VEL.GetX() +
                                            VEL.GetZ() * VEL.GetZ());
                const auto ST = !g_grounded
                    ? CPlayerModel::EState::Jump
                    : HSP > 0.4f ? (g_keySprint ? CPlayerModel::EState::Run
                                                : CPlayerModel::EState::Walk)
                                 : CPlayerModel::EState::Idle;
                g_scene.player()->setState(ST);
            }
        }

        g_msJolt = g_msJolt * 0.9 +
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - JOLT_T0).count() * 0.1;
    }

    g_objTrees.resize(g_sceneObjects.size());
    g_objTreeGens.resize(g_sceneObjects.size(), 0);

    for (size_t i = 0; i < g_sceneObjects.size(); ++i) {
        const uint32_t GEN =
            g_scene.sceneModel(i) ? g_scene.sceneModel(i)->generation() : 0;

        // An AWAKE dynamic Jolt body moves every frame: its collision tree
        // would cost O(n log n) per rebuild, so while it moves the tree is
        // cleared and excluded from player queries; Jolt puts the body to
        // sleep at rest and the tree then builds once.
        const bool AWAKE =
            g_sceneObjects[i].dynamic && g_joltBodies[i].valid && g_bodyIf &&
            g_bodyIf->IsActive(g_joltBodies[i].body);

        const bool SKIP = !g_sceneObjects[i].collision ||
            g_mapGrabIndex == i || AWAKE;

        if (SKIP) {
            // Collision off / carried / still moving: the tree must be EMPTY
            // (the clear belongs to this branch only -- clearing after a
            // build wiped every just-built tree and killed all collision).
            if (!g_objTrees[i].empty())
                g_objTrees[i].clear();
            g_objTreeGens[i] = GEN;
        } else if (g_objTreeGens[i] != GEN) {
            if (const auto* MODEL = g_scene.sceneModel(i))
                g_objTrees[i].build(MODEL->triangles());
            g_objTreeGens[i] = GEN;
        }
    }

    if (g_input.consumeLook(yawDelta, pitchDelta, dt)) {
        g_scene.rotateView(yawDelta, pitchDelta);
        damageCurrentMonitor();
    }

    // C-key view zoom glide: exponential toward the target (the wheel level
    // while held, 1x when released), so both directions are smooth.
    // Sprinting widens the view a little, the common cue for speed.
    const bool SPRINTING = g_keySprint && !g_keyDown &&
        (g_keyFwd || g_keyBack || g_keyLeft || g_keyRight);
    const float ZOOM_TARGET = (g_zoomHeld ? g_zoomWheel : 1.0f) * (SPRINTING ? 0.92f : 1.0f);
    g_zoomLevel += (ZOOM_TARGET - g_zoomLevel) *
        (1.0f - std::exp(-8.0f * dt));
    g_scene.setZoom(g_zoomLevel);

    // ---- player physics body: input -> velocity, Jolt owns the pose ----
    ensurePlayerBody();
    syncFloorBody();
    updateCompanion(dt);
    updatePortals();

    if (!g_playerBody.IsInvalid()) {
        auto& CAM = g_scene.camera();

        // Fell off the world -- past a map's edge, or walking with nothing
        // below: back at the spawn, at rest, instead of falling forever.
        if (g_bodyIf->GetPosition(g_playerBody).GetY() < -80.0f) {
            placeAtSpawn();
            g_bodyIf->SetPositionAndRotation(
                g_playerBody,
                JPH::RVec3(CAM.position.x, CAM.position.y - PLAYER_EYE_OFF, CAM.position.z),
                JPH::Quat::sIdentity(), JPH::EActivation::Activate);
            g_bodyIf->SetLinearVelocity(g_playerBody, JPH::Vec3::sZero());
            s_moveVel = {};
        }

        // collision = false: the capsule joins no contact pair at all --
        // it flies/falls through everything (the honest noclip).
        g_bodyIf->SetObjectLayer(
            g_playerBody,
            g_playerCollision ? LAYER_MOVING : LAYER_GRABBED);

        // The camera was moved by something else (spawn reset, fullscreen
        // transition handoff): teleport the body under it. Skipped while a
        // transition animates (To2D/To3D own the camera then); the
        // transition's end pose resyncs on the first normal frame. First
        // person only: in third person the camera is DERIVED from the body.
        if (g_fsPhase == EFullscreenPhase::None && g_viewMode == 0) {
            const auto CUR = g_bodyIf->GetPosition(g_playerBody);
            const float DX = CAM.position.x - CUR.GetX();
            const float DY = CAM.position.y - (CUR.GetY() + PLAYER_EYE_OFF);
            const float DZ = CAM.position.z - CUR.GetZ();
            if (DX * DX + DY * DY + DZ * DZ > 0.25f)
                g_bodyIf->SetPositionAndRotationWhenChanged(
                    g_playerBody,
                    JPH::RVec3(CAM.position.x, CAM.position.y - PLAYER_EYE_OFF,
                               CAM.position.z),
                    JPH::Quat::sIdentity(), JPH::EActivation::Activate);
        }

        applyCameraMovement(dt); // -> s_moveVel

        // The character's facing: RY(pi - yaw) maps the authored +Z front
        // onto the look direction. Applied AFTER this frame's look update
        // and assigned DIRECTLY -- any smoothing here is second-order on
        // top of the look inertia and reads as a staircase on fast turns.
        {
            const auto PPOS = g_bodyIf->GetPosition(g_playerBody);
            g_scene.setPlayerPose(
                Vec3{PPOS.GetX(), PPOS.GetY() - Camera::kBodyHeight * 0.5f,
                     PPOS.GetZ()},
                3.14159265f - CAM.yaw);

            // The third-person camera SPHERICALLY orbits the player: yaw
            // sweeps the horizontal ring, pitch lifts/drops the camera
            // around the anchor (behind on 0 pitch, above on negative,
            // below on positive), at the same eye level as first person on
            // the zero pitch. Collision-aware: a ray from the head toward
            // the orbit position pulls the camera in front of walls.
            if (g_viewMode != 0) {
                const Vec3 ORBIT_EYE{PPOS.GetX(),
                                     PPOS.GetY() + PLAYER_EYE_OFF,
                                     PPOS.GetZ()};
                const Vec3 FLAT = CAM.flatForward();
                const float P = CAM.pitch;
                Vec3 DIR =
                    FLAT * (-std::cos(P)) + Vec3{0.f, 1.f, 0.f} * (-std::sin(P));
                if (g_viewMode == 2)
                    DIR.x *= -1.f, DIR.y *= -1.f, DIR.z *= -1.f;

                // The distance glides toward the full orbit radius, but is
                // CLAMPED by the ray each frame: the ray probes the FULL
                // orbit distance, and the camera never sits beyond the hit
                // (a hand's width before the wall). Turning changes the hit
                // distance gradually, so the camera slides along walls; the
                // pop-out glides; the push-in can never cross.
                g_playerCamDist += (kThirdDist - g_playerCamDist) *
                    (1.0f - std::exp(-8.0f * dt));

                if (kThirdDist > 1e-4f) {
                    const JPH::RRayCast RAY{
                        JPH::RVec3(ORBIT_EYE.x, ORBIT_EYE.y, ORBIT_EYE.z),
                        JPH::Vec3(DIR.x * kThirdDist, DIR.y * kThirdDist,
                                  DIR.z * kThirdDist)};
                    JPH::RayCastResult HIT;
                    const JPH::IgnoreSingleBodyFilter SKIP_SELF(g_playerBody);
                    if (g_joltSystem->GetNarrowPhaseQuery().CastRay(
                            RAY, HIT, JPH::BroadPhaseLayerFilter(),
                            JPH::ObjectLayerFilter(), SKIP_SELF)) {
                        g_playerCamDist = std::min(
                            g_playerCamDist,
                            std::max(0.05f,
                                     HIT.mFraction * kThirdDist - 0.15f));
                    }
                }
                g_playerCamDist = std::max(g_playerCamDist, 0.05f);

                const Vec3 WANT = ORBIT_EYE + DIR * g_playerCamDist;

                if (WANT.x != CAM.position.x || WANT.y != CAM.position.y ||
                    WANT.z != CAM.position.z) {
                    CAM.position = WANT;
                    damageCurrentMonitor();
                }
            }
        }

        // Flying: all three axes key-driven, gravity asleep. Walking: the
        // keys own the horizontal plane, Jolt's gravity the vertical one.
        if (g_playerFlying) {
            g_bodyIf->SetGravityFactor(g_playerBody, 0.0f);
            g_bodyIf->SetLinearVelocity(g_playerBody, JPH::Vec3(
                s_moveVel.x, s_moveVel.y, s_moveVel.z));
        } else {
            g_bodyIf->SetGravityFactor(g_playerBody, 1.0f);

            auto VEL = g_bodyIf->GetLinearVelocity(g_playerBody);
            float VY = VEL.GetY();
            if (VY < -40.0f) // terminal velocity, as before
                VY = -40.0f;

            // The jump: pressed up to kJumpBuffer before landing, or up to
            // kCoyoteTime after the ground went away. Letting go of Space
            // on the way up halves what is left of the rise.
            g_playerJumpQueued = false;
            const double NOW = nowSeconds();
            if (g_grounded && VY <= 0.5f)
                s_lastGroundedAt = NOW;
            if (s_jumpPressedAt >= 0.0 && NOW - s_jumpPressedAt <= kJumpBuffer &&
                s_lastGroundedAt >= 0.0 && NOW - s_lastGroundedAt <= kCoyoteTime) {
                VY = kJumpSpeed;
                s_jumpPressedAt = s_lastGroundedAt = -1.0;
                s_jumpRising = true;
            }
            if (s_jumpRising && (VY <= 0.0f || !g_keyUp)) {
                if (VY > 0.0f)
                    VY *= 0.5f;
                s_jumpRising = false;
            }

            g_bodyIf->SetLinearVelocity(g_playerBody, JPH::Vec3(
                s_moveVel.x, VY, s_moveVel.z));
        }

        // Walk bob (view-only, walking + grounded): a small vertical sway
        // synced to the distance traveled, plus a head TILT at half the
        // bob frequency (in a real gait the full left-right roll cycle
        // spans two vertical bounces). The amplitude eases in and out, so
        // jumps, stops and the walk<->fly switch never pop.
        const float BOB_SPEED = std::sqrt(s_moveVel.x * s_moveVel.x +
                                          s_moveVel.z * s_moveVel.z);
        const bool BOBING = g_cfgWalkBob && !g_playerFlying && g_grounded &&
            g_fsPhase == EFullscreenPhase::None && g_viewMode == 0;
        const float TARGET_AMP =
            BOBING ? kBobAmplitude *
                std::min(1.0f, BOB_SPEED / std::max(CAM.moveSpeed, 0.5f))
                   : 0.0f;
        s_bobAmp += (TARGET_AMP - s_bobAmp) * (1.0f - std::exp(-8.0f * dt));

        // The phase wraps at 4pi: two bob cycles, so sin(phase/2) walks a
        // FULL tilt cycle (left, then right) and the wrap lands on zero.
        if (BOBING)
            s_bobPhase = std::fmod(s_bobPhase + BOB_SPEED * dt * kBobRate,
                                   12.5663706f);

        // Both offsets ride the same eased envelope: the roll fades out
        // with the bob (walk_bob off, flight, air, transitions).
        float viewY = 0.0f;
        if (s_bobAmp > 0.0005f) {
            viewY += s_bobAmp * std::sin(s_bobPhase);
            CAM.roll = (s_bobAmp * (1.0f / kBobAmplitude)) * kBobRoll *
                std::sin(s_bobPhase * 0.5f);
        } else {
            CAM.roll = 0.0f;
        }

        // Crouch lowers the eye; a landing dips it by how hard it was and
        // springs back. View only, first person, like the bob.
        const bool FIRST = g_fsPhase == EFullscreenPhase::None && g_viewMode == 0;
        const float DROP = FIRST && !g_playerFlying && g_keyDown ? kCrouchDrop : 0.0f;
        s_viewDrop += (DROP - s_viewDrop) * (1.0f - std::exp(-12.0f * dt));
        if (!g_playerFlying && g_grounded && !s_wasGrounded && s_lastFallSpeed > 2.0f)
            s_landDip = std::min(0.12f, s_lastFallSpeed * 0.012f);
        s_landDip *= std::exp(-10.0f * dt);
        s_wasGrounded = g_grounded;
        {
            const auto PV = g_bodyIf->GetLinearVelocity(g_playerBody);
            s_lastFallSpeed = g_grounded ? 0.0f : std::max(0.0f, -PV.GetY());
        }
        if (FIRST)
            viewY -= s_viewDrop + s_landDip;
        CAM.position.y += std::clamp(viewY, -kViewOffsetMax, kViewOffsetMax);

        // Head tracking (g_head): the eye shifted by the head, the view
        // turned by it -- the turn on top of the mouse's, so what was added
        // last frame comes off first.
        headStep(dt);
        const bool  HEAD  = FIRST && g_fsPhase == EFullscreenPhase::None;
        const float YAW   = HEAD ? static_cast<float>(g_head.shown[3] * std::numbers::pi / 180.0) * g_cfgHeadLook : 0.0f;
        const float PITCH = HEAD ? static_cast<float>(g_head.shown[4] * std::numbers::pi / 180.0) * g_cfgHeadLook : 0.0f;
        CAM.yaw += YAW - s_headYawApplied;
        CAM.pitch = std::clamp(CAM.pitch + PITCH - s_headPitchApplied, -1.55f, 1.55f);
        s_headYawApplied = YAW, s_headPitchApplied = PITCH;
        if (HEAD) {
            const float M = g_cfgHeadMove / 100.0f; // cm -> m
            const Vec3  FWD = CAM.flatForward(), RIGHT = CAM.right();
            CAM.position = CAM.position + RIGHT * static_cast<float>(g_head.shown[0] * M) +
                Vec3{0.f, static_cast<float>(g_head.shown[1] * M), 0.f} -
                FWD * static_cast<float>(g_head.shown[2] * M);
        }
    }

    // Map-drag carry: the grabbed object's CENTER rides the crosshair at
    // the pickup distance, and the object turns to face the player (the
    // same billboard convention as window dragging). The new placement is
    // written INTO THE CONFIG OBJECT first: update3D pushes the config
    // specs into the scene every frame, and a carry that only touched the
    // scene slot would be overwritten right back -- the object teleported
    // home on release and both writes recomputed the geometry every frame
    // (the freezes). With the config object as the single source of truth
    // the per-frame push compares equal and skips.
    if (g_pointerGesture == EPointerGesture::MapDrag && g_pointerDown &&
        g_mapGrabIndex != SIZE_MAX) {
        if (g_mapGrabIndex >= g_sceneObjects.size()) {
            g_mapGrabIndex = SIZE_MAX; // config reloaded mid-carry: drop it
        } else {
        const auto& CAM = g_scene.camera();
        const Vec3 TARGET = CAM.position + CAM.centerRay() * g_mapGrabDist;

        // Face the player: the object's +Z normal toward the camera (the
        // model convention maps the normal's Y to -sin(pitch), so pitch =
        // asin(-TO.y) -- the same math as the window drag billboard).
        const Vec3 TO_CAM = normalize(CAM.position - TARGET);
        const float TARGET_YAW = std::atan2(TO_CAM.x, TO_CAM.z);
        const float TARGET_PITCH =
            std::asin(std::clamp(-TO_CAM.y, -1.0f, 1.0f));

        // Smoothly chase the target orientation (wrap-aware on yaw).
        auto& ROT = g_sceneObjects[g_mapGrabIndex].rotationDeg;
        constexpr float DEG = 3.14159265358979f / 180.0f;

        float curYaw = ROT.y * DEG;
        float curPitch = ROT.x * DEG;

        float dYaw = TARGET_YAW - curYaw;
        while (dYaw > 3.14159265f)
            dYaw -= 6.28318531f;
        while (dYaw < -3.14159265f)
            dYaw += 6.28318531f;

        const float K = 1.0f - std::exp(-10.0f * dt);
        curYaw += dYaw * K;
        curPitch += (TARGET_PITCH - curPitch) * K;

        // Full facing on all three axes: the roll eases to zero as well,
        // same exponential as yaw/pitch.
        ROT.z += (0.0f - ROT.z) * K;

        ROT.x = curPitch / DEG;
        ROT.y = curYaw / DEG;

        g_sceneObjects[g_mapGrabIndex].position = TARGET;
        g_scene.setSceneObjectTransform(g_mapGrabIndex, TARGET, ROT);

        if (g_bodyIf && g_mapGrabIndex < g_joltBodies.size() &&
            g_joltBodies[g_mapGrabIndex].valid) {
            const auto JB = g_joltBodies[g_mapGrabIndex].body;
            g_bodyIf->SetLinearVelocity(JB, JPH::Vec3::sZero());
            // Pose-driven while carried: no contact pairs, or the object
            // would shove the player body around on the way.
            g_bodyIf->SetObjectLayer(JB, LAYER_GRABBED);
            g_bodyIf->SetPositionAndRotationWhenChanged(
                JB, JPH::RVec3(TARGET.x, TARGET.y, TARGET.z),
                eulerToQuat(ROT), JPH::EActivation::Activate);
        }

        damageCurrentMonitor();
        }
    }

    // Super+wheel hover zoom: glide the window along its ray toward the
    // target distance. Paused while a fullscreen transition animates (its
    // own glide is one-shot and completes separately).
    if (s_zoomId != 0 && g_fsPhase == EFullscreenPhase::None) {
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

    // Fullscreen-exit box re-assertion: CONDITION-driven, not frame-count --
    // keep re-asserting until the window's box matches ours for 15
    // consecutive frames, outlasting ANY late Hyprland-side restore.
    if (g_fsAssertFrames > 0) {
        if (auto W = g_fsWindow.lock()) {
            Compat::setWindowBox(W, g_fsAssertBox);

            const auto CUR = Compat::currentWindowBox(W);

            if (CUR.x == g_fsAssertBox.x && CUR.y == g_fsAssertBox.y &&
                CUR.w == g_fsAssertBox.w && CUR.h == g_fsAssertBox.h) {
                if (++g_fsStableCount >= 15) {
                    g_fsAssertFrames = 0;
                    g_fsWindow       = {};
                }
            } else {
                g_fsStableCount = 0;
            }
        } else {
            g_fsAssertFrames = 0;
        }
    }

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
    } else if (g_pointerGesture == EPointerGesture::WheelRoll && g_pointerDown) {
        updateWheelRoll();
    }

    gunTick();
    syncWorld(MON, dt);
    useTick();

    applyFullscreenAnimation();

    // Normal client interaction is a virtual pointer located exactly at the
    // crosshair. It is updated every frame after camera motion, so buttons,
    // text fields, scrollbars, etc. receive ordinary Wayland pointer motion.
    if (!g_pointerDown && g_fsPhase == EFullscreenPhase::None)
        forwardPointerToAim(inputTimeMs());
}

// The HUD's quads for the monitor being drawn: its own layers, at their 2D
// place, from their snapshots.
static std::vector<GLScene::SHudQuad> hudQuadsFor(const PHLMONITOR& mon) {
    std::vector<GLScene::SHudQuad> out;
    if (!mon || mon->m_size.x <= 0 || mon->m_size.y <= 0)
        return out;
    for (const auto& H : g_hudItems) {
        if (H.monitor.lock() != mon)
            continue;
        const auto* SNAP = g_capture.get(H.id);
        if (!SNAP || (SNAP->texID == 0 && !SNAP->bigTex) || SNAP->texSpan.x <= 0 || SNAP->texSpan.y <= 0)
            continue;
        const double W = mon->m_size.x, HGT = mon->m_size.y;
        const double SW = SNAP->texSpan.x, SH = SNAP->texSpan.y;
        GLScene::SHudQuad Q;
        Q.texture = SNAP->bigTex ? SNAP->bigTex : SNAP->texID;
        Q.x0 = static_cast<float>(H.box.x / W * 2.0 - 1.0);
        Q.x1 = static_cast<float>((H.box.x + H.box.w) / W * 2.0 - 1.0);
        Q.y0 = static_cast<float>(H.box.y / HGT * 2.0 - 1.0); // -1 = the top
        Q.y1 = static_cast<float>((H.box.y + H.box.h) / HGT * 2.0 - 1.0);
        Q.u0 = static_cast<float>(H.box.x / SW), Q.u1 = static_cast<float>((H.box.x + H.box.w) / SW);
        Q.v0 = static_cast<float>(H.box.y / SH), Q.v1 = static_cast<float>((H.box.y + H.box.h) / SH);
        out.push_back(Q);
    }
    return out;
}

class CHypr3DPassElement final : public IPassElement {
  public:
    // `view` set: a neighbour the room spans -- the same room through its own
    // window of the view plane, drawn without advancing anything.
    CHypr3DPassElement(float alpha, float dt,
                       std::optional<GLScene::ViewWindow> view = std::nullopt) :
        m_alpha(alpha), m_dt(dt), m_view(view) {}

    std::vector<UP<IPassElement>> draw() override {
        const auto GATE = [&](const std::string& why) {
            g_lastRenderGate = why;
            reportFramebufferErrorOnce(why);
            return std::vector<UP<IPassElement>>{};
        };

        if (!g_pHyprRenderer)
            return GATE("[hypr3d] g_pHyprRenderer is null");

        if (g_pHyprRenderer->type() != Render::IHyprRenderer::RT_GL)
            return GATE("[hypr3d] renderer is not OpenGL");

        auto& renderData = g_pHyprRenderer->m_renderData;

        if (!renderData.currentFB)
            return GATE("[hypr3d] current framebuffer is null");

        auto* framebuffer =
            dynamic_cast<Render::GL::CGLFramebuffer*>(
                renderData.currentFB.get()
            );

        if (!framebuffer)
            return GATE("[hypr3d] current framebuffer is not CGLFramebuffer");

        const int width = std::max(
            1,
            static_cast<int>(std::round(renderData.currentFB->m_size.x))
        );

        const int height = std::max(
            1,
            static_cast<int>(std::round(renderData.currentFB->m_size.y))
        );

        const GLuint framebufferID = framebuffer->getFBID();

        if (framebufferID == 0)
            return GATE("[hypr3d] current framebuffer has ID 0");

        if (!Render::GL::g_pHyprOpenGL) {
            reportFramebufferErrorOnce("[hypr3d] g_pHyprOpenGL is null");
            return {};
        }

        Render::GL::g_pHyprOpenGL->makeEGLCurrent();

        g_scene.setHud(hudQuadsFor(renderData.pMonitor.lock()));
        const auto R_T0 = std::chrono::steady_clock::now();
        const bool result = g_scene.render(
            framebufferID,
            width,
            height,
            m_alpha,
            m_dt,
            g_renderWindows,
            m_view ? &*m_view : nullptr,
            !m_view
        );
        const double MS = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - R_T0).count();
        if (m_view)
            g_msRenderSpan = g_msRenderSpan * 0.9 + MS * 0.1;
        else
            g_msRender = g_msRender * 0.9 + MS * 0.1;

        if (!result) {
            if (!g_reportedRenderError) {
                g_reportedRenderError = true;
                notify(
                    "[hypr3d] GLScene::render failed",
                    CHyprColor{1.0f, 0.2f, 0.2f, 1.0f}
                );
            }
            g_lastRenderGate = "GLScene::render returned false";
            return {};
        }

        g_lastRenderGate.clear();

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
    std::optional<GLScene::ViewWindow> m_view;
};

// Throttled snapshot of the live state, written where I can read it directly
// instead of asking the user to describe what they see.
// force: the frame a room goes still -- the last state there is to show,
// which the 0.25 s pace could leave unwritten for as long as it stays still.
static void dumpStatus(bool force = false) {
    const auto NOW = std::chrono::steady_clock::now();

    if (!force && std::chrono::duration<float>(NOW - g_diagLastDump).count() < 0.25f)
        return;

    g_diagLastDump = NOW;
    ++g_diagFrames;

    // Requested here, filled in later this frame by the pass element, read by
    // the next dump -- one frame of lag on a value that is not changing.
    g_scene.requestProbe();

    // History log: APPEND with a timestamp. The one-shot snapshot kept
    // missing the bug (it self-healed before the read); the timeline catches
    // the transition frame by frame. Bounded at ~128 KB, trimmed from the
    // head.
    std::ofstream out("/tmp/hypr3d-status.txt",
                      std::ios::app | std::ios::in);

    {
        std::error_code              ec;
        const auto                   SZ = std::filesystem::file_size(
            "/tmp/hypr3d-status.txt", ec);
        if (!ec && SZ > 128 * 1024) {
            std::ifstream  in("/tmp/hypr3d-status.txt");
            std::string    data((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
            in.close();

            const auto CUT = data.find('\n', data.size() / 2);
            if (CUT != std::string::npos) {
                std::ofstream trim("/tmp/hypr3d-status.txt",
                                   std::ios::trunc);
                trim << data.substr(CUT + 1);
            }
        }
    }

    out << "--- frame " << g_diagFrames << " t="
        << std::chrono::duration<float>(std::chrono::steady_clock::now() -
                                        g_inputClockStart)
               .count()
        << "s\n";

    if (!out)
        return;

    const auto MON = targetMonitor();

    out << "active=" << (g_active ? 1 : 0) << " frames=" << g_diagFrames
        << " transition=" << g_transition
        << " target=" << g_transitionTarget
        << " alphaSent=" << g_diagAlpha
        << " renderedOnce=" << (g_renderedOnce ? 1 : 0) << "\n";

    out << "aimed=" << g_lastAimedId << " focus=" << g_lastFocusId
        << " focusLock=" << g_focusLockId << " aimedConstraint="
        << Compat::pointerConstraintOf(Compat::findWindowById(g_lastAimedId))
        << " gameHasMouse=" << (gameHasMouse() ? 1 : 0) << "\n";

    out << "hookInstalled=" << (g_hookInstalled ? 1 : 0)
        << " hookActive=" << (Compat::pointerHookActive() ? 1 : 0)
        << " sinkCalls=" << g_diagSinkCalls
        << " moveEvents=" << g_diagMoveEvents << "\n";

    out << "lastPos=" << g_diagLastPos.x << "," << g_diagLastPos.y
        << " pinned=" << g_diagPinned.x << "," << g_diagPinned.y
        << " lastDelta=" << g_diagLastDx << "," << g_diagLastDy << "\n";

    out << "lastZoomStep=" << g_diagLastZoomStep
        << " (positive = wheel forward = push away)\n";

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

    // Window-depth pipeline trace: the configured thickness and the
    // silhouette each window draws its walls from (windows: traced from the
    // snapshot texture via the GPU copy, fallback analytic; layers: traced
    // from the picking mask). skirt=0/err=N says the copy failed and why.
    for (const auto& RW : g_renderWindows) {
        out << "  rwin=" << RW.id << " depth=" << RW.depth
            << " outlines=" << (RW.outlines ? (long)RW.outlines->size() : -1);

        if (RW.outlines && !RW.outlines->empty()) {
            const auto& L = (*RW.outlines)[0];
            out << " pts=" << L.pts.size();
            if (!L.pts.empty())
                out << " p0=" << L.pts[0].x << "," << L.pts[0].y;
        }

        if (const auto* SN = g_capture.get(RW.id))
            out << " skirt=" << (SN->skirtValid ? 1 : 0)
                << " err=" << SN->skirtError
                << " mask=" << SN->skirtW << "x" << SN->skirtH;

        out << "\n";
    }

    out << "renderGate=" << (g_lastRenderGate.empty() ? "none" : g_lastRenderGate)
        << "\n";
    out << "renderer=" << (g_pHyprRenderer ? "ok" : "NULL")
        << " rtype=" << (g_pHyprRenderer ? (int)g_pHyprRenderer->type() : -1)
        << " (RT_GL=" << (int)Render::IHyprRenderer::RT_GL << ")\n";
    out << "msUpdate3D=" << g_msUpdate3D << " msJolt=" << g_msJolt
        << " msRender=" << g_msRender << " msRenderSpan=" << g_msRenderSpan << "\n";
    {
        out << "span=" << (g_cfgSpan ? 1 : 0) << " monitors=";
        for (const auto& M : spannedMonitors()) {
            const auto V = viewWindowFor(M);
            out << M->m_name << "[" << V.left << "," << V.right << "," << V.bottom << "," << V.top << "] ";
        }
        out << "\n";
        // What each spanned monitor brings into the room, and how far it got:
        // captured, then an entity of the world -- and where that stands.
        for (const auto& I : eligibleWindowsSpanned()) {
            out << "  item " << (I.isLayer ? "layer" : "window") << " mon="
                << (I.monitor ? I.monitor->m_name : std::string{"?"}) << " box="
                << I.monitorLocalBox.x << "," << I.monitorLocalBox.y << "," << I.monitorLocalBox.w << "x"
                << I.monitorLocalBox.h << " captured=" << (g_capture.has(I.id) ? 1 : 0) << " entity=";
            if (const auto* E = g_world.find(I.id))
                out << "(" << E->center.x << "," << E->center.y << "," << E->center.z << ")\n";
            else
                out << "0\n";
        }
    }
    {
        // Where the player stands and looks, and what the room remembers
        // (see g_room).
        const auto& CAM = g_scene.camera();
        out << "camera: pos=(" << CAM.position.x << "," << CAM.position.y << "," << CAM.position.z
            << ") yaw=" << CAM.yaw << " pitch=" << CAM.pitch << " view=" << g_viewMode
            << " remembered=" << (g_room.valid ? 1 : 0) << "/" << g_room.poses.size() << "\n";
        out << "head: on=" << (g_cfgHead ? 1 : 0) << " port=" << g_head.port << " packets=" << g_head.packets
            << " dropped=" << g_head.dropped << " shown=" << g_head.shown[0] << "," << g_head.shown[1] << ","
            << g_head.shown[2] << " yaw=" << g_head.shown[3] << " pitch=" << g_head.shown[4] << "\n";
        out << "tv: w=" << g_tvW << " window=" << g_tvWindowId << " use=" << g_use.id << " at=" << g_use.local.x
            << "," << g_use.local.y;
        if (const auto* UE = g_use.id ? g_world.find(g_use.id) : nullptr)
            out << " surface=" << UE->surfaceWidth << "x" << UE->surfaceHeight << " box=" << UE->logicalWidth << "x"
                << UE->logicalHeight;
        out << "\n";
        out << "gun: out=" << (g_gun ? 1 : 0) << " holding=" << g_gunHold.id << " shots=" << g_gunShots.size() << "\n";
        out << "read: id=" << g_read.id << " menu=" << g_menuId << " arrived=" << (g_read.arrived ? 1 : 0)
            << " following=" << (g_read.following ? 1 : 0) << " at=(" << g_read.atCenter.x << ","
            << g_read.atCenter.y << "," << g_read.atCenter.z << ") yaw=" << g_read.atYaw
            << " pitch=" << g_read.atPitch << "\n";

        if (!g_companion.body.IsInvalid() && g_bodyIf) {
            static constexpr const char* MODES[] = {"idle", "walk", "turn"};
            const auto P = g_bodyIf->GetPosition(g_companion.body);
            out << "companion: pos=(" << P.GetX() << "," << P.GetY() << "," << P.GetZ()
                << ") yaw=" << g_companion.yaw
                << " mode=" << MODES[static_cast<int>(g_companion.mode)]
                << " target=" << (g_companion.target.empty() ? "-" : g_companion.target)
                << " stuck=" << g_companion.stuck << " room=" << g_roomPosted.substr(0, g_roomPosted.find('\n'))
                << "\n";
        }
    }
    out << "lastError=" << (g_lastError.empty() ? "none" : g_lastError)
        << "\n";
    out << "sceneObjects=" << g_sceneObjects.size() << " joltBodies="
        << g_joltBodies.size() << " objTrees=" << g_objTrees.size() << "\n";

    out << "fsPhase=" << static_cast<int>(g_fsPhase)
        << " fsWasOn=" << (g_fsWasOn ? 1 : 0)
        << " fsAlpha=" << g_fsAlpha << "\n";

    if (const auto FSW = Fullscreen::controller()->getFullscreenWindow(
            targetMonitor())) {
        const auto BOX = Compat::currentWindowBox(FSW);
        float fsRoll = 0.0f;
        if (auto* E = g_world.find(Compat::windowId(FSW)))
            fsRoll = E->roll;

        out << "fsWindow=" << Compat::windowId(FSW)
            << " box=" << BOX.x << "," << BOX.y << "," << BOX.w << "," << BOX.h
            << " restore=" << g_fsRestoreBox.w << "x" << g_fsRestoreBox.h
            << " roll=" << fsRoll << "\n";

    } else {
        out << "fsWindow=none\n";
    }
    // Scene-object physics state: one line per object.
    for (size_t i = 0; i < g_sceneObjects.size() && i < g_objTrees.size(); ++i) {
        const auto& OBJ = g_sceneObjects[i];
        const auto& J = i < g_joltBodies.size() ? g_joltBodies[i]
                                                : SObjJolt{};
        out << "  obj" << i << "="
            << OBJ.path.substr(OBJ.path.size() -
                               std::min<size_t>(OBJ.path.size(), 24))
            << " dyn=" << (OBJ.dynamic ? 1 : 0)
            << " phys=" << (OBJ.physics ? 1 : 0)
            << " col=" << (OBJ.collision ? 1 : 0)
            << " pos=" << OBJ.position.x << "," << OBJ.position.y << ","
            << OBJ.position.z
            << " tree=" << (g_objTrees[i].empty() ? 0 : 1)
            << " jolt=" << (J.valid ? 1 : 0)
            << "\n";
    }

    // Per-window channel alphas + Hyprland's remembered floating size:
    // the semi-transparency and the giant-size bugs live in THESE state
    // channels; the dump pins which one is stuck and for whom.
    if (Desktop::windowState() && MON && MON->m_activeWorkspace) {
        for (auto const& W3 : Desktop::windowState()->windows()) {
            if (!W3 || W3->m_workspace != MON->m_activeWorkspace)
                continue;

            const auto WB = W3->m_target ? W3->m_target->position() :
                                           CBox{};
            out << "  win=" << Compat::windowId(W3)
                << " float=" << (W3->m_isFloating ? 1 : 0)
                << " aFull="
                << W3->alphaValue(Desktop::View::WINDOW_ALPHA_FULLSCREEN)
                << " aActive="
                << W3->alphaValue(Desktop::View::WINDOW_ALPHA_ACTIVE)
                << " aFade="
                << W3->alphaValue(Desktop::View::WINDOW_ALPHA_FADE)
                << " box=" << WB.x << "," << WB.y << "," << WB.w << ","
                << WB.h << " lastFloat="
                << (W3->m_target ? W3->m_target->lastFloatingSize().x : -1.f)
                << "x"
                << (W3->m_target ? W3->m_target->lastFloatingSize().y : -1.f)
                << "\n";
        }
    }

    out << "captureFrames=" << g_captureFrames << " still=" << g_stillFrames
        << " onTime=" << (runsOnTime() ? 1 : 0) << "\n";

    // Player body trace: wall glue, stick-slip or a resync fight show up
    // here directly (position/velocity vs the commanded velocity).
    if (!g_playerBody.IsInvalid() && g_bodyIf) {
        const auto PP = g_bodyIf->GetPosition(g_playerBody);
        const auto PV = g_bodyIf->GetLinearVelocity(g_playerBody);
        out << "player: pos=(" << PP.GetX() << "," << PP.GetY() << ","
            << PP.GetZ() << ") vel=(" << PV.GetX() << "," << PV.GetY()
            << "," << PV.GetZ() << ") moveVel=(" << s_moveVel.x << ","
            << s_moveVel.y << "," << s_moveVel.z << ") grounded="
            << g_grounded << " flying=" << g_playerFlying << "\n";
    }
}

// --- rendering on demand ----------------------------------------------------
// The room drew every frame at the monitor's rate, standing still too: 552
// frames in 10 s, 28 % of a core and 23 % of the RTX 3080 for a picture that
// did not change (measured 2026-10-07) -- on a laptop, the battery. As
// Godot's low-processor mode and three.js's render-on-demand do, it asks for
// the next frame only while something changes: each frame folds what makes
// the picture into a fingerprint, and after kStillFrames identical ones, with
// nothing running on time, it stops asking. Input asks for a frame itself
// (every handler damages), a window's new content through Hyprland's own
// damage, and the frame pump looks for time-driven work 20 times a second.
// (State and the pump's interval: before the pump.)
static uint64_t frameFingerprint() {
    uint64_t   h = 1469598103934665603ull;
    const auto BITS = [&](uint64_t v) {
        h ^= v;
        h *= 1099511628211ull;
    };
    // Quantized: a body at rest still moves by nanometres a step.
    const auto NUM = [&](double v, double quantum) {
        BITS(static_cast<uint64_t>(std::llround(v / quantum)));
    };
    const auto POS = [&](float x, float y, float z) {
        NUM(x, 1e-4), NUM(y, 1e-4), NUM(z, 1e-4);
    };
    const auto& CAM = g_scene.camera();
    POS(CAM.position.x, CAM.position.y, CAM.position.z);
    NUM(CAM.yaw, 1e-5), NUM(CAM.pitch, 1e-5);
    BITS(static_cast<uint64_t>(g_viewMode));
    NUM(g_zoomLevel, 1e-4);
    BITS(g_world.entities().size());
    for (const auto& E : g_world.entities()) {
        BITS(E.id);
        POS(E.center.x, E.center.y, E.center.z);
        NUM(E.yaw, 1e-5), NUM(E.pitch, 1e-5), NUM(E.roll, 1e-5);
        NUM(E.width, 1e-4), NUM(E.height, 1e-4);
    }
    BITS(g_scene.sceneFingerprint());
    if (g_bodyIf) {
        for (const auto& J : g_joltBodies) {
            if (!J.valid)
                continue;
            const auto P = g_bodyIf->GetPosition(J.body);
            const auto R = g_bodyIf->GetRotation(J.body);
            POS(P.GetX(), P.GetY(), P.GetZ());
            NUM(R.GetX(), 1e-5), NUM(R.GetY(), 1e-5), NUM(R.GetZ(), 1e-5), NUM(R.GetW(), 1e-5);
        }
        if (!g_companion.body.IsInvalid()) {
            const auto P = g_bodyIf->GetPosition(g_companion.body);
            POS(P.GetX(), P.GetY(), P.GetZ());
            NUM(g_companion.yaw, 1e-5);
        }
    }
    BITS(g_read.id);
    NUM(g_read.dim, 1e-3);
    BITS(g_use.id); // F8's pointer is drawn
    NUM(g_use.local.x, 0.5), NUM(g_use.local.y, 0.5);
    BITS(static_cast<uint64_t>(g_fsPhase));
    NUM(g_fsAlpha, 1e-3), NUM(g_transition, 1e-3), NUM(g_diagAlpha, 1e-3);
    return h;
}

// What moves on its own, with no state yet to show it: an animation clock,
// work done off the main thread, a request waiting for the next frame.
static bool runsOnTime() {
    return g_debugHud                        // the HUD counts frames
        || g_viewMode != 0                   // F5: the player's own animation
        || g_scene.scenePending() || joltShapesPending()
        || g_sightPending
        || g_companion.mode != ECompanionMode::Idle
        || !g_trashed.empty()
        || (g_read.id && (!g_read.arrived || g_read.following || g_read.back))
        || g_world.dragActive()
        || !g_gunShots.empty() || g_gunHold.active
        || headBusy()
        || g_transition != g_transitionTarget
        || g_fsPhase == EFullscreenPhase::To2D || g_fsPhase == EFullscreenPhase::To3D;
}

static bool roomStill() {
    return g_stillFrames >= kStillFrames && !runsOnTime();
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

    if (!g_currentRenderMon)
        return;

    if (g_currentRenderMon != g_monitor) {
        // A neighbour the room spans: the room as the target's frame left
        // it -- no clock, no physics, no transition step of its own -- through
        // this monitor's window of the view plane.
        if (g_fsPhase == EFullscreenPhase::In2D || g_transition <= 0.0f ||
            !g_pHyprRenderer || g_pHyprRenderer->type() != Render::IHyprRenderer::RT_GL)
            return;

        g_pHyprRenderer->addPassElement(makeUnique<CHypr3DPassElement>(
            g_diagAlpha, 0.0f, viewWindowFor(g_currentRenderMon)));
        return;
    }

    dumpStatus();

    // Locked: not one more frame of the room, and closed without the fade
    // (the fade draws the room). The teardown runs from the event loop.
    if (sessionLocked()) {
        g_transitionTarget = 0.0f;
        g_transition       = 0.0f;
        requestDeactivate3D();
        syncRoomState(false);
        abortSight("The session is locked");
        return;
    }

    syncRoomState(false);
    deliverSight();

    const float dt = updateTransition();

    if (g_transition <= 0.0f && g_transitionTarget <= 0.0f) {
        requestDeactivate3D();
        return;
    }

    if (!g_pHyprRenderer)
        return;

    if (g_pHyprRenderer->type() != Render::IHyprRenderer::RT_GL)
        return;

    // 2D passthrough: Hyprland renders the fullscreen window; the room and
    // its input are dormant. The frame pump polls for the fullscreen exit.
    if (g_fsPhase == EFullscreenPhase::In2D)
        return;

    const auto U3_T0 = std::chrono::steady_clock::now();
    update3D(dt);
    g_msUpdate3D = g_msUpdate3D * 0.9 +
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - U3_T0).count() * 0.1;

    g_diagAlpha = std::clamp(g_transition, 0.0f, 1.0f);

    if (g_fsPhase == EFullscreenPhase::To2D || g_fsPhase == EFullscreenPhase::To3D)
        g_diagAlpha = g_fsAlpha;

    g_pHyprRenderer->addPassElement(
        makeUnique<CHypr3DPassElement>(
            g_diagAlpha,
            dt
        )
    );

    const uint64_t SIG = frameFingerprint();
    g_stillFrames = SIG == g_frameSig && !runsOnTime() ? g_stillFrames + 1 : 0;
    g_frameSig = SIG;
    if (!roomStill())
        damageCurrentMonitor();
    else if (g_stillFrames == kStillFrames)
        dumpStatus(/*force=*/true);
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
    damageCurrentMonitor(); // a still room draws again (roomStill)

    if (event.axis != WL_POINTER_AXIS_VERTICAL_SCROLL)
        return;

    // Physical wheels report whole detents in deltaDiscrete; touchpads send
    // a smooth delta instead. Hi-res wheels report many detents per physical
    // click -- clamped, or a light scroll flings the window across the room.
    //
    // Sign, per the Wayland axis spec: a vertical delta is negative when the
    // top of the wheel rolls AWAY from the user ("forward") with
    // IDENTICAL direction, and positive when the client is told INVERTED
    // (natural scroll). Normalize to +1 step = wheel forward in both cases,
    // so the zoom below can be written against the physical wheel motion
    // instead of guessing the compositor's sign.
    const double RAW = event.deltaDiscrete != 0 ?
        static_cast<double>(event.deltaDiscrete) :
        event.delta * 0.05;

    const bool INVERTED = event.relativeDirection ==
        WL_POINTER_AXIS_RELATIVE_DIRECTION_INVERTED;

    double STEPS = std::clamp(INVERTED ? RAW : -RAW, -2.0, 2.0);

    if (STEPS == 0.0)
        return;

    g_diagLastZoomStep = STEPS;

    // C-held view zoom: the wheel adjusts the magnification live
    // (min 1x, no max). Wheel forward = zoom in.
    if (g_zoomHeld) {
        g_zoomWheel = std::max(
            1.0f, g_zoomWheel * static_cast<float>(std::pow(1.06, STEPS)));
        info.cancelled = true;
        damageCurrentMonitor();
        return;
    }

    // During an LMB drag the wheel zooms the dragged window.
    if (g_pointerGesture == EPointerGesture::Move3D && g_pointerDown) {
        g_world.dragZoom(STEPS);
        info.cancelled = true;
        damageCurrentMonitor();
        return;
    }

    // Carrying a scene object: the wheel scales the grab distance, so the
    // object approaches or recedes along the crosshair. (modelRayHit skips
    // the carried object -- it is under the crosshair already -- so without
    // this branch the wheel would do nothing during a carry.)
    if (g_pointerGesture == EPointerGesture::MapDrag && g_pointerDown &&
        g_mapGrabIndex != SIZE_MAX) {
        g_mapGrabDist = std::max(
            0.05f, g_mapGrabDist * static_cast<float>(std::pow(1.06, STEPS)));
        info.cancelled = true;
        damageCurrentMonitor();
        return;
    }

    // Any other time: Super + wheel over a window zooms it along the ray
    // from the camera through the window. Without Super the wheel reaches
    // the focused client unchanged (scroll).
    if (!g_superHeld)
        return;

    // On the TV it scales the window instead: forward, its text larger.
    if (g_tvWindowId) {
        const auto HIT = aimHit();
        if (const auto W = HIT.hit && HIT.id == g_tvWindowId ? Compat::findWindowById(HIT.id) : nullptr) {
            tvResize(W, Compat::currentWindowBox(W).h * std::pow(1.12, -STEPS));
            info.cancelled = true;
            damageCurrentMonitor();
            return;
        }
    }

    // A DYNAMIC scene object closer than any window zooms instead: its
    // center slides along the camera-object line, same multiplicative step,
    // same no-limits policy as the window zoom.
    {
        const auto& CAMZ = g_scene.camera();
        const auto MRAY = modelRayHit(CAMZ.position, CAMZ.centerRay(), true);
        const World3D::SHit WHIT = aimHit();

        const bool MODEL_CLOSER =
            MRAY.hit && (!WHIT.hit || MRAY.dist < WHIT.distance);

        if (!MODEL_CLOSER) {
            if (s_modelZoomId != SIZE_MAX)
                s_modelZoomId = SIZE_MAX; // aimed away: drop the zoom
        } else {
            const auto* MODEL = g_scene.sceneModel(MRAY.index);

            if (!MODEL || !MODEL->loaded()) {
                s_modelZoomId = SIZE_MAX;
                return;
            }

            if (s_modelZoomId != MRAY.index)
                s_modelZoomDist = std::sqrt(
                    (MODEL->position() - CAMZ.position).x *
                        (MODEL->position() - CAMZ.position).x +
                    (MODEL->position() - CAMZ.position).y *
                        (MODEL->position() - CAMZ.position).y +
                    (MODEL->position() - CAMZ.position).z *
                        (MODEL->position() - CAMZ.position).z);
            s_modelZoomId = MRAY.index;

            const float NEW_DIST = s_modelZoomDist * std::pow(1.06, STEPS);
            const Vec3 DIR = normalize(MODEL->position() - CAMZ.position);

            g_sceneObjects[MRAY.index].position =
                CAMZ.position + DIR * NEW_DIST;
            g_scene.setSceneObjectTransform(MRAY.index,
                                            g_sceneObjects[MRAY.index].position,
                                            MODEL->rotationDeg());

            if (g_bodyIf && MRAY.index < g_joltBodies.size() &&
                g_joltBodies[MRAY.index].valid)
                g_bodyIf->SetPositionAndRotationWhenChanged(
                    g_joltBodies[MRAY.index].body,
                    JPH::RVec3(g_sceneObjects[MRAY.index].position.x,
                               g_sceneObjects[MRAY.index].position.y,
                               g_sceneObjects[MRAY.index].position.z),
                    eulerToQuat(g_sceneObjects[MRAY.index].rotationDeg),
                    JPH::EActivation::Activate);

            s_modelZoomDist = NEW_DIST;

            info.cancelled = true;
            damageCurrentMonitor();
            return;
        }
    }

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

    // Wheel forward (+) pushes the window away, wheel back pulls it closer;
    // the drag zoom receives the same normalized sign, so both wheel paths
    // agree. No distance limits -- the target stays positive, so scrolling
    // just keeps multiplying; the window can come arbitrarily close or far.
    s_zoomTarget *= std::pow(1.06, STEPS);

    info.cancelled = true;
    damageCurrentMonitor();
}

// What the gun aims at: the first window along the crosshair's ray. Not a
// layer, and not Hyprland's own dialog: when a program hangs, Hyprland puts
// "Application Not Responding" over it, and the gun aims through it at the
// program it asks about (measured: the shot hit the dialog).
static PHLWINDOW gunTarget() {
    const auto& CAM = g_scene.camera();
    for (const auto& HIT : g_world.pickAll(CAM.position, CAM.centerRay())) {
        if (!hitVisible(HIT))
            continue;
        const auto T = targetFromHit(HIT.id);
        if (!T.window || CAsyncDialogBox::isAsyncDialogBox(T.window->getPID()))
            continue;
        return T.window;
    }
    return nullptr;
}

// The process gun's trigger (g_gun): the window under the crosshair is shot.
static void gunFire() {
    const auto WINDOW = gunTarget();
    if (!WINDOW)
        return; // a layer or the void: nothing to shoot
    const auto ID  = Compat::windowId(WINDOW);
    const auto NOW = std::chrono::steady_clock::now();
    std::erase_if(g_gunShots, [&](const SGunShot& S) { return S.id == ID; });
    g_gunShots.push_back({ID, NOW, g_scene.camera().centerRay()});
    g_gunHold = {true, ID, NOW};
    if (g_pEventLoopManager)
        g_pEventLoopManager->doLater([ID] {
            if (const auto W = Compat::findWindowById(ID))
                W->sendClose();
        });
}

// Every frame: shots that are done, and the kill being held -- dropped if
// the crosshair left the window, carried out when held long enough.
static void gunTick() {
    const auto NOW = std::chrono::steady_clock::now();
    std::erase_if(g_gunShots, [&](const SGunShot& S) {
        return std::chrono::duration<float>(NOW - S.at).count() >= kGunJoltSeconds;
    });
    float charge = 0.0f;
    bool  hung   = false;
    if (g_gun && g_gunHold.active) {
        const auto WINDOW = gunTarget();
        if (!WINDOW || Compat::windowId(WINDOW) != g_gunHold.id)
            g_gunHold = {};
        else {
            hung   = g_pANRManager && g_pANRManager->isNotResponding(WINDOW);
            charge = std::chrono::duration<float>(NOW - g_gunHold.since).count() / kGunKillSeconds;
            if (charge >= 1.0f) {
                const auto ID = g_gunHold.id;
                g_gunHold = {};
                charge    = 0.0f;
                if (g_pEventLoopManager)
                    g_pEventLoopManager->doLater([ID] {
                        const auto W = Compat::findWindowById(ID);
                        const pid_t PID = W ? W->getPID() : 0;
                        if (PID > 1 && PID != getpid()) {
                            ::kill(PID, SIGKILL);
                            notify("[hypr3d] killed " + W->m_title, CHyprColor{1.0f, 0.3f, 0.2f, 1.0f});
                        }
                    });
            }
        }
    }
    g_scene.setGunSight(g_gun, charge, hung);
}

static void onMouseButton(
    IPointer::SButtonEvent event,
    Event::SCallbackInfo& info
) {
    if (!ownsInput())
        return;
    damageCurrentMonitor(); // a still room draws again (roomStill)

    const bool PRESSED =
        event.state == WL_POINTER_BUTTON_STATE_PRESSED;

    // A window in use takes the buttons where its pointer is.
    if (g_use.id) {
        if (const auto W = Compat::findWindowById(g_use.id))
            Compat::deliverClick(W, g_use.local, event.button, PRESSED, inputTimeMs());
        info.cancelled = true;
        return;
    }

    // The process gun owns the left button while it is out.
    if (g_gun && event.button == BTN_LEFT) {
        info.cancelled = true;
        if (PRESSED)
            gunFire();
        else
            g_gunHold = {};
        return;
    }

    // Release the plugin gesture that owns this physical button.
    if (!PRESSED && g_pointerDown && event.button == g_pointerButton) {
        if (g_pointerGesture == EPointerGesture::Move3D && !dropInBin(g_world.draggedId()))
            dropOnTV(g_world.draggedId());
        resetPointerGesture();
        info.cancelled = true;
        damageCurrentMonitor();
        return;
    }

    // Super + wheel-press: grab the aimed window's roll around its normal.
    // Sweeping the crosshair around the window center then rotates the
    // window by the swept angle; release ends it. Never reaches the client.
    if (PRESSED && g_superHeld && g_fsPhase == EFullscreenPhase::None &&
        event.button == BTN_MIDDLE) {
        const World3D::SHit HIT = aimHit();
        const auto* ENTITY = HIT.hit ? g_world.find(HIT.id) : nullptr;

        // A dynamic scene object closer than any window: roll IT. The roll
        // axis is its facing normal; the sweep angle is measured on the
        // plane through its center, exactly like the window roll.
        {
            const auto& CAM = g_scene.camera();
            const auto MRAY = modelRayHit(CAM.position, CAM.centerRay(),
                                          /*dynamicOnly=*/true);

            if (MRAY.hit && (!HIT.hit || MRAY.dist < HIT.distance)) {
                const auto* MODEL = g_scene.sceneModel(MRAY.index);
                const Vec3 CENTER = MODEL->position();
                const Vec3 NORMAL = normalize(CAM.position - CENTER);

                Vec3 PLANE_POINT;
                if (rayPlanePoint(CAM.position, CAM.centerRay(), CENTER,
                                  NORMAL, PLANE_POINT) &&
                    dot(PLANE_POINT - CENTER, PLANE_POINT - CENTER) > 0.0004f) {
                    s_wheelRot.active     = true;
                    s_wheelRot.model      = true;
                    s_wheelRot.modelIndex = MRAY.index;
                    s_wheelRot.id         = 0;
                    s_wheelRot.center     = CENTER;
                    s_wheelRot.normal     = NORMAL;
                    s_wheelRot.reference  = PLANE_POINT - CENTER;
                    s_wheelRot.startRoll  = MODEL->rotationDeg().z *
                        (3.14159265358979f / 180.0f);

                    // Capture the gravity factor ONCE: the per-frame update
                    // sets it to 0 for the suspension, and re-reading it
                    // there would store the zero and never restore it.
                    if (MRAY.index < g_joltBodies.size() &&
                        g_joltBodies[MRAY.index].valid)
                        g_rolledGravity = g_bodyIf->GetGravityFactor(
                            g_joltBodies[MRAY.index].body);

                    g_pointerGesture = EPointerGesture::WheelRoll;
                    g_pointerButton  = BTN_MIDDLE;
                    g_pointerDown    = true;
                    g_resize         = {};

                    info.cancelled = true;
                    damageCurrentMonitor();
                    return;
                }

                info.cancelled = true;
                return;
            }
        }

        if (ENTITY) {
            const Vec3 REFERENCE = HIT.point - ENTITY->center;

            // The sweep angle is undefined when the crosshair sits exactly
            // on the center -- refuse the grab there.
            if (dot(REFERENCE, REFERENCE) > 0.0004f) {
                s_wheelRot.active     = true;
                s_wheelRot.id         = HIT.id;
                s_wheelRot.center     = ENTITY->center;
                s_wheelRot.normal     = g_world.normalOf(HIT.id);
                s_wheelRot.reference  = REFERENCE;
                s_wheelRot.startRoll  = ENTITY->roll;

                g_pointerGesture = EPointerGesture::WheelRoll;
                g_pointerButton  = BTN_MIDDLE;
                g_pointerDown    = true;
                g_resize         = {};

                info.cancelled = true;
                damageCurrentMonitor();
                return;
            }
        }

        info.cancelled = true;
        return;
    }

    // Super+LMB / Super+RMB are exclusively plugin gestures. Nothing is sent
    // to the client for these physical button events. Layer surfaces can be
    // dragged like windows, but not resized -- their real geometry is owned
    // by the shell that anchored them.
    if (PRESSED && g_superHeld && g_fsPhase == EFullscreenPhase::None &&
        (event.button == BTN_LEFT || event.button == BTN_RIGHT)) {
        const World3D::SHit HIT = aimHit();
        const auto TARGET = HIT.hit ? targetFromHit(HIT.id) : SHitTarget{};

        // A DYNAMIC scene object closer than any window is grabbed with the
        // left button (one class of grabbable objects: windows and models).
        // The grabbed object leaves the collision set for the duration (it
        // follows the crosshair and must not push the player), and returns
        // to it on release.
        if (event.button == BTN_LEFT) {
            const auto& CAM = g_scene.camera();
            const Vec3 DIR = CAM.centerRay();

            float bestT = -1.f;
            size_t bestIdx = SIZE_MAX;

            for (size_t i = 0; i < g_sceneObjects.size(); ++i) {
                if (!g_sceneObjects[i].dynamic)
                    continue;

                const auto* MODEL = g_scene.sceneModel(i);
                if (!MODEL || !MODEL->loaded())
                    continue;

                const float T = MODEL->rayCast(CAM.position, DIR);
                if (T > 0.f && (bestT < 0.f || T < bestT)) {
                    bestT   = T;
                    bestIdx = i;
                }
            }

            // The model must also be CLOSER than the aimed window -- one
            // distance space for both classes.
            if (bestIdx != SIZE_MAX && bestT < 0.f)
                bestIdx = SIZE_MAX;
            if (bestIdx != SIZE_MAX && HIT.hit && bestT >= HIT.distance)
                bestIdx = SIZE_MAX;

            if (bestIdx != SIZE_MAX) {
                const auto* MODEL = g_scene.sceneModel(bestIdx);

                // Center grab: the object's center rides the crosshair, so
                // the pickup distance is the camera-to-CENTER distance and
                // there is no point offset.
                g_mapGrabIndex = bestIdx;
                g_mapGrabDist  = std::sqrt(
                    (MODEL->position() - CAM.position).x *
                        (MODEL->position() - CAM.position).x +
                    (MODEL->position() - CAM.position).y *
                        (MODEL->position() - CAM.position).y +
                    (MODEL->position() - CAM.position).z *
                        (MODEL->position() - CAM.position).z);

                s_modelZoomId = SIZE_MAX; // the drag owns the object now

                g_pointerGesture = EPointerGesture::MapDrag;
                g_pointerButton  = BTN_LEFT;
                g_pointerDown    = true;
                g_resize         = {};

                info.cancelled = true;
                damageCurrentMonitor();
                return;
            }
        }

        if (!TARGET.window && !TARGET.layer) {
            info.cancelled = true;
            return;
        }

        const auto& CAM = g_scene.camera();

        if (TARGET.window && g_focusLockId == 0)
            Compat::focusWindow(TARGET.window);

        // Resize is a window-only control: a scene model in front of the
        // crosshair must not let the gesture reach a window behind it.
        if (event.button == BTN_RIGHT) {
            if (!TARGET.window || modelInFront(CAM.position,
                                               CAM.centerRay(), HIT)) {
                info.cancelled = true;
                return;
            }
        }

        if (event.button == BTN_LEFT) {
            if (!g_world.startDrag(
                    HIT.id, HIT, CAM.position, CAM.forward())) {
                info.cancelled = true;
                return;
            }
            s_zoomId = 0; // the drag owns this window's distance now
            if (HIT.id == g_tvWindowId)
                g_tvWindowId = 0; // carried off the TV

            // A drag on the assert window: the drag wins, stop fighting.
            if (g_fsAssertFrames > 0 && HIT.id == Compat::windowId(TARGET.window)) {
                g_fsAssertFrames = 0;
                g_fsWindow = {};
            }

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

            // The grab point only seeds which edge follows the crosshair;
            // updateRealResize re-evaluates that every frame from the aim's
            // side relative to the window centre, so the grab lands anywhere
            // on the window and the pull direction decides the rest. As the
            // camera turns, the current centre ray is intersected with this
            // same window plane, so the real window stretches exactly toward
            // the point being aimed at.
            // RayHit::v is already top-to-bottom. Top half follows +1,
            // bottom half follows -1 in the CBox edge convention below.
            g_resize.edgeX = HIT.u < 0.5f ? -1 : 1;
            g_resize.edgeY = HIT.v < 0.5f ? 1 : -1;
            g_resize.grabPx = worldPointToGlobalPx(targetMonitor(), HIT.point);

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
    // camera turns away before release. A scene model closer than the aimed
    // window occludes it: the click hits geometry, not the client.
    const auto& CAMC = g_scene.camera();
    const World3D::SHit HIT = aimHit();

    if (modelInFront(CAMC.position, CAMC.centerRay(), HIT)) {
        info.cancelled = true;
        return;
    }

    const auto TARGET = HIT.hit ? targetFromHit(HIT.id) : SHitTarget{};

    if (!TARGET.window && !TARGET.layer) {
        info.cancelled = true;
        return;
    }

    const Vector2D LOCAL = localFromHit(HIT);

    if (PRESSED) {
        // Locked, a click reaches the window under the crosshair and the
        // keyboard stays where it was locked.
        if (TARGET.window && g_focusLockId == 0)
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
    // Every input wakes a still room (roomStill): a key changes its state
    // in the next frame, and with no frame asked for there was none.
    damageCurrentMonitor();

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

    // A game holding the mouse has the keyboard too; keys held for walking
    // let go, so the player does not walk on behind it.
    if (gameHasMouse()) {
        resetMovementKeys();
        return;
    }

    // Walking mode: Space jumps off whatever the capsule stands on. The
    // held state still reaches setMovementSym, but the walking movement
    // path zeroes the vertical input, so holding Space does not fly.
    // SPACE KEYBOARD MODE ONLY: in Window mode the key belongs to the
    // focused window -- a jump here would fire on every typed space.
    if (PRESSED && SYM == XKB_KEY_space &&
        g_keyboardMode == EKeyboardMode::Space) {
        const double NOW = nowSeconds();
        if (s_lastSpacePressAt >= 0.0 && NOW - s_lastSpacePressAt <= kDoubleTap) {
            // The second press of a double tap toggles flight; flying starts
            // hovering where it is, walking falls from there.
            g_playerFlying = !g_playerFlying;
            s_lastSpacePressAt = s_jumpPressedAt = -1.0;
            s_jumpRising = false;
            s_moveVel.y = 0.0f;
            if (g_playerFlying && !g_playerBody.IsInvalid() && g_bodyIf) {
                const auto V = g_bodyIf->GetLinearVelocity(g_playerBody);
                g_bodyIf->SetLinearVelocity(g_playerBody, JPH::Vec3(V.GetX(), 0.f, V.GetZ()));
            }
        } else {
            s_lastSpacePressAt = NOW;
            if (!g_playerFlying)
                s_jumpPressedAt = NOW; // the jump itself happens in update3D
        }
    }

    // F3 toggles the debug HUD (collision wireframe + info overlay) in both
    // keyboard modes: it never belongs to the focused window.
    if (PRESSED && SYM == XKB_KEY_F3) {
        g_debugHud = !g_debugHud;
        g_scene.setMapDebugCollisions(g_debugHud);
        g_scene.setDebugOverlay(g_debugHud);

        info.cancelled = true;
        damageCurrentMonitor();
        return;
    }

    // F8: use the window under the crosshair, or stop (g_use).
    if (PRESSED && SYM == XKB_KEY_F8) {
        if (g_use.id)
            useEnd();
        else
            useBegin();
        info.cancelled = true;
        return;
    }

    // F7: the process gun out, or away (g_gun); Escape puts it away too.
    if (PRESSED && (SYM == XKB_KEY_F7 || (g_gun && SYM == XKB_KEY_Escape))) {
        g_gun     = SYM == XKB_KEY_F7 ? !g_gun : false;
        g_gunHold = {};
        info.cancelled = true;
        return;
    }

    // F6 locks the keyboard on the window that has it, or releases it, in
    // both keyboard modes -- like F3 it never belongs to the focused window.
    if (PRESSED && SYM == XKB_KEY_F6) {
        setFocusLock(g_focusLockId == 0);
        info.cancelled = true;
        return;
    }

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

    // F1: the room's menu, as VRChat's (config menu.command).
    if (PRESSED && SYM == XKB_KEY_F1) {
        if (g_cfgMenuCommand.empty())
            notify("[hypr3d] F1: no menu set (config menu.command)", CHyprColor{0.2f, 0.8f, 0.4f, 1.0f});
        else if (!Config::Supplementary::executor()->spawn(g_cfgMenuCommand))
            notify("[hypr3d] F1: the menu command did not start", CHyprColor{1.0f, 0.2f, 0.2f, 1.0f});
        info.cancelled = true;
        return;
    }

    // F2: the aimed window to the eye at 1:1, and back. F4: the cinema.
    if (PRESSED && (SYM == XKB_KEY_F2 || SYM == XKB_KEY_F4)) {
        toggleReading(SYM == XKB_KEY_F4);
        info.cancelled = true;
        return;
    }

    // F5: cycle the view -- first person, third person behind, third
    // person in front.
    if (PRESSED && SYM == XKB_KEY_F5) {
        g_viewMode = (g_viewMode + 1) % 3;
        info.cancelled = true;
        damageCurrentMonitor();
        return;
    }

    if (SYM == XKB_KEY_c) {
        // View zoom: hold to magnify, release to ease back to 1x. The
        // wheel level restarts at the base on every press.
        g_zoomHeld = PRESSED;
        if (PRESSED)
            g_zoomWheel = kZoomBase;
        info.cancelled = true;
        return;
    }

    if (!isMovementSym(SYM))
        return;

    setMovementSym(SYM, PRESSED);

    info.cancelled = true;
}

// --- plugin entry -----------------------------------------------------------

static int luaConfig(lua_State* L) {
    // hl.plugin.hypr3d.config({
    //     world = {
    //         panorama = "~/picture.png",   -- 360-degree room background
    //         grid = true,                  -- base grid platform (visible +
    //                                       -- collidable, at world zero)
    //         shadows = true,               -- a soft shadow on the floor
    //                                       -- under each window
    //         hud = true,                   -- the rice's bar and
    //                                       -- notifications over the view,
    //                                       -- as in 2D (false: on the wall)
    //     },
    //     windows = {
    //         window_scale = 0.5,           -- room multiplier on window size
    //         spawn_distance = 5.0,         -- units in front of the camera
    //         depth = 0.05,                 -- window slab thickness, world
    //                                       -- units (0 = flat quads; walls
    //                                       -- follow rounded corners and
    //                                       -- show the texture's edge)
    //     },
    //     player = {
    //         look_sensitivity = 0.0025,    -- radians per pointer count
    //         look_inertia = 0.03,          -- look glide, seconds (0 = off)
    //         move_inertia = 0.05,          -- walk glide, seconds (0 = off)
    //         move_speed = 4.0,             -- world units / second
    //         spawn = { x = 0, y = 0, z = 0 }, -- FEET position
    //         flying = true,                -- false: gravity, Space jumps,
    //                                       -- Shift does nothing
    //         walk_bob = true,              -- camera sway while walking
    //     },
    //     map = {
    //         path = "~/map.glb",
    //         transform = {
    //             position = { x = 0, y = 0, z = 0 },
    //             rotation = { x = 0, y = 0, z = 0 }, -- degrees, XYZ
    //             scale = { x = 1, y = 1, z = 1 },    -- per-axis
    //         },
    //         emissive_scale = 1.0,
    //         flat = true,                  -- baked-map look (no dynamic light)
    //         collision = true,
    //     },
    // })
    //
    // Missing keys keep their current value; wrong-typed keys raise a lua
    // error.
    if (!lua_istable(L, 1))
        return luaL_error(L, "hypr3d.config expects a single table");

    // Field accessors. TIDX = stack index of the section table (0 = absent).
    const auto SET_NUM = [&](int tidx, const char* key, float& out,
                             const char* path) -> bool {
        lua_getfield(L, tidx, key);
        if (lua_isnil(L, -1)) {
            lua_pop(L, 1);
            return true;
        }
        if (!lua_isnumber(L, -1)) {
            lua_pop(L, 1);
            return false;
        }
        out = static_cast<float>(lua_tonumber(L, -1));
        lua_pop(L, 1);
        return true;
    };

    const auto SET_BOOL = [&](int tidx, const char* key, bool& out,
                              const char* path) -> bool {
        lua_getfield(L, tidx, key);
        if (lua_isnil(L, -1)) {
            lua_pop(L, 1);
            return true;
        }
        if (!lua_isboolean(L, -1)) {
            lua_pop(L, 1);
            return false;
        }
        out = lua_toboolean(L, -1) != 0;
        lua_pop(L, 1);
        return true;
    };

    const auto SET_STRING = [&](int tidx, const char* key, std::string& out,
                                const char* path) -> bool {
        lua_getfield(L, tidx, key);
        if (lua_isnil(L, -1)) {
            lua_pop(L, 1);
            return true;
        }
        if (!lua_isstring(L, -1)) {
            lua_pop(L, 1);
            return false;
        }
        size_t LEN = 0;
        const char* STR = lua_tolstring(L, -1, &LEN);
        out.assign(STR, LEN);
        lua_pop(L, 1);
        return true;
    };

    // Missing axes keep their current values.
    const auto SET_VEC3 = [&](int tidx, const char* key, Vec3& out,
                              const char* path) -> bool {
        lua_getfield(L, tidx, key);
        if (lua_isnil(L, -1)) {
            lua_pop(L, 1);
            return true;
        }
        if (!lua_istable(L, -1)) {
            lua_pop(L, 1);
            return false;
        }

        const int T = lua_gettop(L);

        // Positional form: { x, y, z } == { 1, 2, 3 }.
        float pos[3] = {0.f, 0.f, 0.f};
        bool havePos = false;
        for (int k = 1; k <= 3; ++k) {
            lua_rawgeti(L, T, k);
            if (lua_isnil(L, -1)) {
                lua_pop(L, 1);
                continue;
            }
            if (!lua_isnumber(L, -1)) {
                lua_pop(L, 1);
                return false;
            }
            pos[k - 1] = static_cast<float>(lua_tonumber(L, -1));
            havePos = true;
            lua_pop(L, 1);
        }

        const auto AXIS = [&](const char* name, float& v) {
            lua_getfield(L, T, name);
            if (lua_isnumber(L, -1))
                v = static_cast<float>(lua_tonumber(L, -1));
            lua_pop(L, 1);
        };

        float x = out.x, y = out.y, z = out.z;
        AXIS("x", x);
        AXIS("y", y);
        AXIS("z", z);

        // Positional wins when both forms are mixed in one table.
        if (havePos)
            out = Vec3{pos[0], pos[1], pos[2]};
        else
            out = Vec3{x, y, z};

        lua_pop(L, 1);
        return true;
    };

    // A sub-table section: returns its stack index, or 0 when absent/wrong.
    // A wrong-typed section is an error, a missing one is skipped.
    const auto SECTION = [&](const char* key, const char* path) -> int {
        lua_getfield(L, 1, key);
        if (lua_isnil(L, -1)) {
            lua_pop(L, 1);
            return 0;
        }
        if (!lua_istable(L, -1))
            return -1; // caller reports
        return lua_gettop(L);
    };

    int idx = SECTION("world", "world");
    if (idx == -1)
        return luaL_error(L, "hypr3d.config: world must be a table");
    if (idx > 0) {
        if (!SET_STRING(idx, "panorama", g_cfgPanorama, "world.panorama"))
            return luaL_error(L, "hypr3d.config: world.panorama must be a string");
        if (!SET_STRING(idx, "monitor", g_cfgMonitor, "world.monitor"))
            return luaL_error(L, "hypr3d.config: world.monitor must be a string");
        if (!SET_BOOL(idx, "grid", g_cfgGrid, "world.grid"))
            return luaL_error(L, "hypr3d.config: world.grid must be a boolean");
        if (!SET_BOOL(idx, "shadows", g_cfgShadows, "world.shadows"))
            return luaL_error(L, "hypr3d.config: world.shadows must be a boolean");
        if (!SET_BOOL(idx, "hud", g_cfgHud, "world.hud"))
            return luaL_error(L, "hypr3d.config: world.hud must be a boolean");
        if (!SET_BOOL(idx, "span", g_cfgSpan, "world.span"))
            return luaL_error(L, "hypr3d.config: world.span must be a boolean");
        if (!SET_NUM(idx, "gravity", g_cfgGravity, "world.gravity"))
            return luaL_error(L, "hypr3d.config: world.gravity must be a number");
        g_cfgGravity = std::clamp(g_cfgGravity, 0.5f, 40.0f);
        if (g_joltSystem)
            g_joltSystem->SetGravity(JPH::Vec3(0.f, -g_cfgGravity, 0.f));
        lua_pop(L, 1);
    }

    idx = SECTION("windows", "windows");
    if (idx == -1)
        return luaL_error(L, "hypr3d.config: windows must be a table");
    if (idx > 0) {
        if (!SET_NUM(idx, "window_scale", g_cfgWindowScale,
                     "windows.window_scale"))
            return luaL_error(L, "hypr3d.config: windows.window_scale must be a number");
        if (!SET_NUM(idx, "spawn_distance", g_cfgSpawnDistance,
                     "windows.spawn_distance"))
            return luaL_error(L, "hypr3d.config: windows.spawn_distance must be a number");
        if (!SET_NUM(idx, "depth", g_cfgWindowDepth,
                     "windows.depth"))
            return luaL_error(L, "hypr3d.config: windows.depth must be a number");

        // Thickness is a distance: 0 (the default) draws the flat quads,
        // anything below is clamped up to it.
        g_cfgWindowDepth = std::max(0.0f, g_cfgWindowDepth);

        lua_pop(L, 1);
    }

    // trash = { at = { x, y, z }, radius = 0.3, height = 0.75 }: the waste
    // bin's zone, its foot at `at`; radius 0 takes it away.
    idx = SECTION("trash", "trash");
    if (idx == -1)
        return luaL_error(L, "hypr3d.config: trash must be a table");
    if (idx > 0) {
        if (!SET_VEC3(idx, "at", g_trashAt, "trash.at"))
            return luaL_error(L, "hypr3d.config: trash.at must be a vector");
        if (!SET_NUM(idx, "radius", g_trashRadius, "trash.radius"))
            return luaL_error(L, "hypr3d.config: trash.radius must be a number");
        if (!SET_NUM(idx, "height", g_trashHeight, "trash.height"))
            return luaL_error(L, "hypr3d.config: trash.height must be a number");
        g_trashRadius = std::clamp(g_trashRadius, 0.0f, 5.0f);
        g_trashHeight = std::clamp(g_trashHeight, 0.0f, 5.0f);
        lua_pop(L, 1);
    }

    idx = SECTION("tv", "tv");
    if (idx == -1)
        return luaL_error(L, "hypr3d.config: tv must be a table");
    if (idx > 0) {
        if (!SET_VEC3(idx, "at", g_tvAt, "tv.at") || !SET_NUM(idx, "yaw", g_tvYawDeg, "tv.yaw") ||
            !SET_NUM(idx, "width", g_tvW, "tv.width") || !SET_NUM(idx, "height", g_tvH, "tv.height"))
            return luaL_error(L, "hypr3d.config: tv = { at, yaw, width, height } is invalid");
        g_tvW = std::clamp(g_tvW, 0.0f, 20.0f);
        g_tvH = std::clamp(g_tvH, 0.0f, 20.0f);
        if (g_tvW <= 0.0f)
            g_tvWindowId = 0;
        lua_pop(L, 1);
    }

    // menu = { command = "...", title = "..." }: F1 in the room runs the
    // command; a window with this title comes to the eye as it opens.
    idx = SECTION("menu", "menu");
    if (idx == -1)
        return luaL_error(L, "hypr3d.config: menu must be a table");
    if (idx > 0) {
        if (!SET_STRING(idx, "command", g_cfgMenuCommand, "menu.command"))
            return luaL_error(L, "hypr3d.config: menu.command must be a string");
        if (!SET_STRING(idx, "title", g_cfgMenuTitle, "menu.title"))
            return luaL_error(L, "hypr3d.config: menu.title must be a string");
        lua_pop(L, 1);
    }

    idx = SECTION("head", "head");
    if (idx == -1)
        return luaL_error(L, "hypr3d.config: head must be a table");
    if (idx > 0) {
        if (!SET_BOOL(idx, "enabled", g_cfgHead, "head.enabled") ||
            !SET_NUM(idx, "port", g_cfgHeadPort, "head.port") ||
            !SET_NUM(idx, "look", g_cfgHeadLook, "head.look") ||
            !SET_NUM(idx, "move", g_cfgHeadMove, "head.move"))
            return luaL_error(L, "hypr3d.config: head = { enabled, port, look, move } is invalid");
        lua_pop(L, 1);
        headApply();
    }

    idx = SECTION("player", "player");
    if (idx == -1)
        return luaL_error(L, "hypr3d.config: player must be a table");
    if (idx > 0) {
        if (!SET_NUM(idx, "look_sensitivity", g_cfgSensitivity,
                     "player.look_sensitivity"))
            return luaL_error(L, "hypr3d.config: player.look_sensitivity must be a number");
        if (!SET_NUM(idx, "look_inertia", g_cfgLookInertia,
                     "player.look_inertia"))
            return luaL_error(L, "hypr3d.config: player.look_inertia must be a number");
        if (!SET_NUM(idx, "move_inertia", g_cfgMoveInertia,
                     "player.move_inertia"))
            return luaL_error(L, "hypr3d.config: player.move_inertia must be a number");
        if (!SET_NUM(idx, "move_speed", g_cfgMoveSpeed,
                     "player.move_speed"))
            return luaL_error(L, "hypr3d.config: player.move_speed must be a number");
        if (!SET_BOOL(idx, "flying", g_playerFlying, "player.flying"))
            return luaL_error(L, "hypr3d.config: player.flying must be a boolean");
        if (!SET_BOOL(idx, "walk_bob", g_cfgWalkBob, "player.walk_bob"))
            return luaL_error(L, "hypr3d.config: player.walk_bob must be a boolean");
        if (!SET_BOOL(idx, "collision", g_playerCollision, "player.collision"))
            return luaL_error(L, "hypr3d.config: player.collision must be a boolean");

        // The shared mesh-block parser: path + transform + material
        // overrides -- the SAME description for the player and the scene
        // objects. transform.position anchors the mesh, rotation.y corrects
        // the authored facing, scale stretches it.
        const auto PARSE_MESH = [&](int MT, std::string& path, Vec3& pos,
                                    Vec3& rotDeg, Vec3& scale,
                                    float& emissive, bool& flat,
                                    std::string& center,
                                    Vec3& centerOff) -> bool {
            if (!SET_STRING(MT, "path", path, "mesh.path"))
                return false;

            lua_getfield(L, MT, "transform");
            if (lua_isnil(L, -1)) {
                lua_pop(L, 1);
            } else if (!lua_istable(L, -1)) {
                lua_pop(L, 1);
                return false;
            } else {
                const int TIDX = lua_gettop(L);
                if (!SET_VEC3(TIDX, "position", pos, "mesh.transform.position") ||
                    !SET_VEC3(TIDX, "rotation", rotDeg, "mesh.transform.rotation") ||
                    !SET_VEC3(TIDX, "scale", scale, "mesh.transform.scale"))
                    return false;
                lua_pop(L, 1);
            }

            if (!SET_NUM(MT, "emissive_scale", emissive, "mesh.emissive_scale") ||
                !SET_BOOL(MT, "flat", flat, "mesh.flat") ||
                !SET_STRING(MT, "center", center, "mesh.center") ||
                !SET_VEC3(MT, "center_offset", centerOff, "mesh.center_offset"))
                return false;
            return true;
        };

        // player.mesh: the character's visual, described exactly like a
        // scene object's mesh. The transform.position anchors the model
        // relative to the feet, rotation.y corrects the authored facing.
        lua_getfield(L, idx, "mesh");
        if (lua_isnil(L, -1)) {
            lua_pop(L, 1);
        } else if (!lua_istable(L, -1)) {
            lua_pop(L, 1);
            return luaL_error(L, "hypr3d.config: player.mesh must be a table");
        } else {
            const int MT = lua_gettop(L);
            if (!PARSE_MESH(MT, g_playerCfg.path, g_playerCfg.posOffset,
                            g_playerCfg.rotDeg, g_playerCfg.scale,
                            g_playerCfg.emissiveScale, g_playerCfg.flat,
                            g_playerCfg.center, g_playerCfg.centerOffset))
                return luaL_error(L, "hypr3d.config: player.mesh is invalid");
            lua_pop(L, 1);
        }

        // Legacy single keys, kept as a fallback for older configs. Applied
        // ONLY when the key is actually present: model_scale used to rebuild
        // the scale as {x, x, x} on every parse, silently collapsing the
        // mesh block's per-axis transform.scale.
        lua_getfield(L, idx, "model");
        if (!lua_isnil(L, -1)) {
            lua_pop(L, 1);
            if (!SET_STRING(idx, "model", g_playerCfg.path, "player.model"))
                return luaL_error(L, "hypr3d.config: player.model must be a string");
        } else {
            lua_pop(L, 1);
        }
        lua_getfield(L, idx, "model_scale");
        if (!lua_isnil(L, -1)) {
            lua_pop(L, 1);
            float SCALE_F = g_playerCfg.scale.x;
            if (!SET_NUM(idx, "model_scale", SCALE_F, "player.model_scale"))
                return luaL_error(L, "hypr3d.config: player.model_scale must be a number");
            g_playerCfg.scale = Vec3{SCALE_F, SCALE_F, SCALE_F};
        } else {
            lua_pop(L, 1);
        }
        lua_getfield(L, idx, "model_turn");
        if (!lua_isnil(L, -1)) {
            lua_pop(L, 1);
            if (!SET_NUM(idx, "model_turn", g_playerCfg.rotDeg.y, "player.model_turn"))
                return luaL_error(L, "hypr3d.config: player.model_turn must be a number");
        } else {
            lua_pop(L, 1);
        }

        const auto SET_ANIM = [&](const char* key, int slot) -> bool {
            lua_getfield(L, idx, key);
            if (lua_isnil(L, -1)) {
                lua_pop(L, 1);
                return true;
            }
            if (lua_isnumber(L, -1)) {
                g_playerCfg.animIdx[slot] = static_cast<int>(lua_tonumber(L, -1));
                g_playerCfg.animName[slot].clear();
            } else if (lua_isstring(L, -1)) {
                size_t LEN = 0;
                const char* STR = lua_tolstring(L, -1, &LEN);
                g_playerCfg.animName[slot].assign(STR, LEN);
                g_playerCfg.animIdx[slot] = -1;
            } else {
                lua_pop(L, 1);
                return false;
            }
            lua_pop(L, 1);
            return true;
        };
        if (!SET_ANIM("anim_idle", 0) || !SET_ANIM("anim_walk", 1) ||
            !SET_ANIM("anim_run", 2) || !SET_ANIM("anim_jump", 3))
            return luaL_error(L, "hypr3d.config: player.anim_* must be an animation index or name");

        // animations = { idle = {source = "Idle" | 0, duration_scale = 1.0},
        //                ... } -- source is a clip name or an index;
        // duration_scale is the playback SPEED multiplier (1 = as authored).
        static const char* const STATE_KEYS[4] = {"idle", "walk", "run", "jump"};
        lua_getfield(L, idx, "animations");
        if (lua_isnil(L, -1)) {
            lua_pop(L, 1);
        } else if (!lua_istable(L, -1)) {
            lua_pop(L, 1);
            return luaL_error(L, "hypr3d.config: player.animations must be a table");
        } else {
            const int AT = lua_gettop(L);
            for (int slot = 0; slot < 4; ++slot) {
                lua_getfield(L, AT, STATE_KEYS[slot]);
                if (lua_isnil(L, -1)) {
                    lua_pop(L, 1);
                    continue;
                }
                if (!lua_istable(L, -1)) {
                    lua_pop(L, 1);
                    return luaL_error(L,
                        "hypr3d.config: player.animations.%s must be a table",
                        STATE_KEYS[slot]);
                }
                const int ST = lua_gettop(L);

                lua_getfield(L, ST, "source");
                if (lua_isnumber(L, -1)) {
                    g_playerCfg.animIdx[slot] = static_cast<int>(lua_tonumber(L, -1));
                    g_playerCfg.animName[slot].clear();
                } else if (lua_isstring(L, -1)) {
                    size_t LEN = 0;
                    const char* STR = lua_tolstring(L, -1, &LEN);
                    g_playerCfg.animName[slot].assign(STR, LEN);
                    g_playerCfg.animIdx[slot] = -1;
                } else {
                    lua_pop(L, 2);
                    return luaL_error(L,
                        "hypr3d.config: player.animations.%s.source must be a clip name or index",
                        STATE_KEYS[slot]);
                }
                lua_pop(L, 1); // source

                float SPEED = 1.0f;
                if (!SET_NUM(ST, "duration_scale", SPEED,
                             "player.animations.<state>.duration_scale"))
                    return luaL_error(L,
                        "hypr3d.config: player.animations.<state>.duration_scale must be a number");
                g_playerAnimSpeed[slot] = SPEED;
                g_playerCfg.animSpeed[slot] = SPEED;

                lua_pop(L, 1); // the state table
            }
            lua_pop(L, 1); // animations
        }
        if (!SET_VEC3(idx, "spawn", g_playerSpawn, "player.spawn"))
            return luaL_error(L, "hypr3d.config: player.spawn must be a table { x = .., y = .., z = .. }");
        lua_pop(L, 1);
    }

    // scene = { <any name> = { path, transform, emissive_scale, flat,
    //           collision, static }, ... } -- unlimited named objects.
    // Object names are free-form (for the user's readability only).
    idx = SECTION("scene", "scene");
    if (idx == -1)
        return luaL_error(L, "hypr3d.config: scene must be a table");
    if (idx > 0) {
        std::vector<SSceneObjectCfg> OBJECTS;

        lua_pushnil(L);
        while (lua_next(L, idx) != 0) {
            // stack: [key, value]
            if (!lua_istable(L, -1)) {
                lua_pop(L, 1); // keep the key for the next iteration
                continue;
            }

            SSceneObjectCfg OBJ;
            const int OIDX = lua_gettop(L);

            // The table key names the object. Reconciliation matches by
            // NAME, not index: lua_next order is NOT stable across config
            // parses (string hashes are seeded per lua state, so two parses
            // of the SAME file can iterate the objects in opposite orders),
            // and an index-matched reconcile cross-wired the objects -- the
            // untouched static map inherited a dynamic object's fallen pose
            // and rotated away out of view.
            if (lua_type(L, OIDX - 1) == LUA_TSTRING) {
                size_t KLEN = 0;
                const char* KSTR = lua_tolstring(L, OIDX - 1, &KLEN);
                OBJ.name.assign(KSTR, KLEN);
            }

            if (!SET_STRING(OIDX, "path", OBJ.path, "scene.<name>.path")) {
                lua_pop(L, 1);
                return luaL_error(L, "hypr3d.config: scene.<name>.path must be a string");
            }

            // transform = { position, rotation, scale } -- a NESTED table.
            // (An earlier parser revision looked for position/rotation/scale
            // directly on the object, found nothing, and every object
            // rendered at its identity transform.)
            lua_getfield(L, OIDX, "transform");
            if (lua_isnil(L, -1)) {
                lua_pop(L, 1);
            } else if (!lua_istable(L, -1)) {
                lua_pop(L, 1);
                return luaL_error(L, "hypr3d.config: scene.<name>.transform must be a table with position/rotation/scale");
            } else {
                const int TIDX = lua_gettop(L);

                if (!SET_VEC3(TIDX, "position", OBJ.position,
                              "scene.<name>.transform.position"))
                    return luaL_error(L, "hypr3d.config: scene.<name>.transform.position must be a table { x = .., y = .., z = .. }");
                if (!SET_VEC3(TIDX, "rotation", OBJ.rotationDeg,
                              "scene.<name>.transform.rotation"))
                    return luaL_error(L, "hypr3d.config: scene.<name>.transform.rotation must be a table { x = .., y = .., z = .. } (degrees)");
                if (!SET_VEC3(TIDX, "scale", OBJ.scale,
                              "scene.<name>.transform.scale"))
                    return luaL_error(L, "hypr3d.config: scene.<name>.transform.scale must be a table { x = .., y = .., z = .. }");

                lua_pop(L, 1);
            }

            if (!SET_NUM(OIDX, "emissive_scale", OBJ.emissiveScale,
                         "scene.<name>.emissive_scale")) {
                lua_pop(L, 1);
                return luaL_error(L, "hypr3d.config: scene.<name>.emissive_scale must be a number");
            }
            if (!SET_BOOL(OIDX, "flat", OBJ.flat, "scene.<name>.flat")) {
                lua_pop(L, 1);
                return luaL_error(L, "hypr3d.config: scene.<name>.flat must be a boolean");
            }
            if (!SET_BOOL(OIDX, "collision", OBJ.collision, "scene.<name>.collision")) {
                lua_pop(L, 1);
                return luaL_error(L, "hypr3d.config: scene.<name>.collision must be a boolean");
            }

            // static = true (default): immovable location geometry.
            // static = false: a dynamic object -- grabbable with Super+LMB.
            bool staticObj = true;
            if (!SET_BOOL(OIDX, "static", staticObj, "scene.<name>.static")) {
                lua_pop(L, 1);
                return luaL_error(L, "hypr3d.config: scene.<name>.static must be a boolean");
            }
            OBJ.dynamic = !staticObj;

            // physics = true: gravity + world collisions (dynamic only;
            // ignored on static objects).
            if (!SET_BOOL(OIDX, "physics", OBJ.physics, "scene.<name>.physics")) {
                lua_pop(L, 1);
                return luaL_error(L, "hypr3d.config: scene.<name>.physics must be a boolean");
            }

            // Rotation pivot: "logical" rotates around the mesh's local AABB
            // center, "origin" around its own origin; center_offset shifts
            // the pivot on top in local units.
            std::string CENTER = "logical";
            if (!SET_STRING(OIDX, "center", CENTER, "scene.<name>.center")) {
                lua_pop(L, 1);
                return luaL_error(L, "hypr3d.config: scene.<name>.center must be a string");
            }
            OBJ.center = CENTER == "origin" ? CMapModel::ECenter::Origin
                                            : CMapModel::ECenter::Logical;
            if (!SET_VEC3(OIDX, "center_offset", OBJ.centerOffset,
                          "scene.<name>.center_offset")) {
                lua_pop(L, 1);
                return luaL_error(L, "hypr3d.config: scene.<name>.center_offset must be a table { x = .., y = .., z = .. }");
            }

            // mesh = { ... }: the shared visual description -- overrides
            // the legacy flat keys when present.
            lua_getfield(L, OIDX, "mesh");
            if (lua_istable(L, -1)) {
                const int MT = lua_gettop(L);
                std::string CENTER = "origin";
                Vec3 MROT{};
                bool MFLAT = OBJ.flat;
                float MEMIS = OBJ.emissiveScale;
                if (!SET_STRING(MT, "path", OBJ.path, "mesh.path") ||
                    !SET_VEC3(MT, "center_offset", OBJ.centerOffset,
                              "mesh.center_offset") ||
                    !SET_NUM(MT, "emissive_scale", MEMIS, "mesh.emissive_scale") ||
                    !SET_BOOL(MT, "flat", MFLAT, "mesh.flat") ||
                    !SET_STRING(MT, "center", CENTER, "mesh.center"))
                    return luaL_error(L, "hypr3d.config: scene.<name>.mesh is invalid");
                OBJ.emissiveScale = MEMIS;
                OBJ.flat = MFLAT;
                OBJ.center = CENTER == "origin"
                    ? CMapModel::ECenter::Origin : CMapModel::ECenter::Logical;

                lua_getfield(L, MT, "transform");
                if (lua_isnil(L, -1)) {
                    lua_pop(L, 1);
                } else if (!lua_istable(L, -1)) {
                    lua_pop(L, 1);
                    return luaL_error(L, "hypr3d.config: scene.<name>.mesh.transform must be a table");
                } else {
                    const int TIDX = lua_gettop(L);
                    if (!SET_VEC3(TIDX, "position", OBJ.position,
                                  "mesh.transform.position") ||
                        !SET_VEC3(TIDX, "rotation", OBJ.rotationDeg,
                                  "mesh.transform.rotation") ||
                        !SET_VEC3(TIDX, "scale", OBJ.scale,
                                  "mesh.transform.scale"))
                        return luaL_error(L, "hypr3d.config: scene.<name>.mesh.transform is invalid");
                    lua_pop(L, 1);
                }
            }
            lua_pop(L, 1); // pop mesh (or its nil)

            OBJECTS.push_back(OBJ);
            lua_pop(L, 1); // pop the value; the key remains for lua_next
        }

        // Canonical order: lua_next order varies between parses (seeded
        // hashes), and every index-keyed structure downstream (GLScene
        // slots, Jolt bodies, BVH trees) would churn -- or swap model files
        // between slots -- on every reload. Sorting by name makes the
        // parsed list deterministic.
        std::sort(OBJECTS.begin(), OBJECTS.end(),
                  [](const SSceneObjectCfg& A, const SSceneObjectCfg& B) {
                      return A.name < B.name;
                  });

        // Snapshot the RAW parsed lua FIRST: this is the file's truth, and
        // the next reload compares against it to tell "user edited the file"
        // from "the simulation moved the object". (Snapshotting after the
        // reconciliation below would record simulated positions as config
        // values -- the following reload then saw a phantom change and kept
        // teleporting carried/fallen objects back to the config placement.)
        g_sceneLuaState = OBJECTS;
        g_sceneLuaValid = true;

        // Reconciliation, per field, matched BY NAME: a field whose lua
        // value did not change keeps the simulated state (carry / zoom /
        // roll / physics); a field edited in the file takes the new config
        // value. Scale is never simulated, so it always comes from the
        // config. Static objects are config-owned by definition -- they
        // never move by simulation and never take simulated state, so a
        // past parse glitch cannot persist through them either.
        for (auto& NEW : OBJECTS) {
            if (!NEW.dynamic)
                continue;

            const SSceneObjectCfg* OLD = nullptr;
            for (const auto& O : g_sceneLuaState)
                if (O.name == NEW.name) {
                    OLD = &O;
                    break;
                }
            if (!OLD)
                continue;

            const SSceneObjectCfg* SIM = nullptr;
            for (const auto& O : g_sceneObjects)
                if (O.name == NEW.name) {
                    SIM = &O;
                    break;
                }
            if (!SIM)
                continue;

            if (OLD->position.x == NEW.position.x &&
                OLD->position.y == NEW.position.y &&
                OLD->position.z == NEW.position.z)
                NEW.position = SIM->position;

            if (OLD->rotationDeg.x == NEW.rotationDeg.x &&
                OLD->rotationDeg.y == NEW.rotationDeg.y &&
                OLD->rotationDeg.z == NEW.rotationDeg.z)
                NEW.rotationDeg = SIM->rotationDeg;
        }

        g_sceneObjects = std::move(OBJECTS);
        lua_pop(L, 1);
    }

    g_scene.setPlayerConfig(g_playerCfg);
    for (int slot = 0; slot < 4; ++slot)
        g_scene.player()->setAnimSpeed(static_cast<CPlayerModel::EState>(slot),
                                       g_playerAnimSpeed[slot]);

    return 0;
}

// Back to the spawn point with every window on the wall, as the first time:
// the room forgets how it was left.
static void resetRoom() {
    g_room = {};
    g_read = {};
    g_trashed.clear();
    g_wallFit = 0.0f; // the wall is built afresh, sized for the camera then
    companionHalt("aborted", "The room was reset");
    placeCompanion();
    if (!g_active)
        return;

    g_scene.reset();
    placeAtSpawn();

    // In first person the camera is read back from the player's body every
    // frame, so a camera moved alone snaps back: the body goes too, at rest.
    if (!g_playerBody.IsInvalid() && g_bodyIf) {
        const auto& CAM = g_scene.camera();
        g_bodyIf->SetPositionAndRotation(
            g_playerBody,
            JPH::RVec3(CAM.position.x, CAM.position.y - PLAYER_EYE_OFF, CAM.position.z),
            JPH::Quat::sIdentity(), JPH::EActivation::Activate);
        g_bodyIf->SetLinearVelocity(g_playerBody, JPH::Vec3::sZero());
    }

    g_world.clear();
    g_viewMode = 0;
    damageCurrentMonitor();
}

static int luaReset(lua_State*) {
    resetRoom();
    return 0;
}

// hl.plugin.hypr3d.companion(verb[, target]): go_to and look_at check the
// target and set off, or fail with the reason -- hyprctl eval prints it and
// exits non-zero. stop halts; state posts the room event now, for a bridge
// that has just connected. The checks run in a function of their own that
// returns before lua_error: Lua's error is a longjmp, and it must not skip
// the destructors of live C++ objects.
static bool companionLua(lua_State* L) {
    const char* VERB = lua_type(L, 1) == LUA_TSTRING ? lua_tostring(L, 1) : "";
    const std::string verb = VERB;
    if (verb == "state" || verb == "stop" || verb == "go_to" || verb == "look_at" ||
        verb == "see")
        g_companionCalled = true;

    if (verb == "see") {
        const long long NOW = monotonicNs();
        if (!roomOpen())
            lua_pushstring(L, "companion: the room is closed");
        else if (g_sightPending)
            lua_pushstring(L, "companion: the last picture is still on its way");
        else if (NOW - g_sightAskedNs < 1'000'000'000LL) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "companion: one picture a second; ask again in %.1f s",
                          (1e9 - static_cast<double>(NOW - g_sightAskedNs)) / 1e9);
            lua_pushstring(L, buf);
        } else {
            g_sightAsked   = true;
            g_sightPending = true;
            g_sightAskedNs = NOW;
            return true;
        }
        return false;
    }

    if (verb == "state") {
        postRoomState();
        return true;
    }
    if (verb == "stop") {
        companionHalt(nullptr, {});
        return true;
    }
    if (verb != "go_to" && verb != "look_at") {
        lua_pushstring(L, "companion: the verb is go_to, look_at, see, stop or state");
        return false;
    }
    if (lua_type(L, 2) != LUA_TSTRING) {
        lua_pushstring(L, "companion: go_to and look_at need a target name");
        return false;
    }

    // Names are lower case; what the model typed may not be.
    std::string target = lua_tostring(L, 2);
    for (auto& c : target)
        if (c >= 'A' && c <= 'Z')
            c = c - 'A' + 'a';
    const auto FIRST = target.find_first_not_of(' ');
    target = FIRST == std::string::npos ?
        "" : target.substr(FIRST, target.find_last_not_of(' ') - FIRST + 1);
    if (target.size() > 64)
        target.resize(64);

    std::string why;
    if (companionStart(verb == "go_to", target, why))
        return true;
    lua_pushstring(L, ("companion: " + why).c_str());
    return false;
}

// hl.plugin.hypr3d.portal(name, { image = "/path.png", command = "...",
//     at = { x, y, z } or front = true, yaw = degrees, width = metres })
// sets or replaces the portal `name`; hl.plugin.hypr3d.portal(name) removes
// it. `front = true` stands it 2.5 m in front of the player, facing him;
// `yaw` defaults to facing the spawn. A wrong argument fails with the reason,
// before lua_error, as companionLua does.
static bool portalLua(lua_State* L) {
    if (lua_type(L, 1) != LUA_TSTRING) {
        lua_pushstring(L, "portal: the first argument is its name");
        return false;
    }
    const std::string NAME = lua_tostring(L, 1);
    const auto IT = std::ranges::find_if(g_portals, [&](const SPortal& p) { return p.spec.name == NAME; });
    if (lua_isnoneornil(L, 2)) {
        if (IT == g_portals.end()) {
            lua_pushstring(L, ("portal: there is no portal '" + NAME + "'").c_str());
            return false;
        }
        g_portals.erase(IT);
        syncPortals();
        return true;
    }
    if (!lua_istable(L, 2)) {
        lua_pushstring(L, "portal: the second argument is a table, or nothing to remove it");
        return false;
    }

    const auto STRING = [&](const char* key) -> std::string {
        lua_getfield(L, 2, key);
        std::string out = lua_type(L, -1) == LUA_TSTRING ? lua_tostring(L, -1) : "";
        lua_pop(L, 1);
        return out;
    };
    const auto NUMBER = [&](const char* key, float fallback) -> float {
        lua_getfield(L, 2, key);
        const float out = lua_isnumber(L, -1) ? static_cast<float>(lua_tonumber(L, -1)) : fallback;
        lua_pop(L, 1);
        return out;
    };

    SPortal portal;
    portal.spec.name  = NAME;
    portal.spec.image = STRING("image");
    portal.command    = STRING("command");
    portal.spec.width = std::clamp(NUMBER("width", 1.8f), 0.3f, 10.0f);
    if (portal.spec.image.empty() || !std::filesystem::is_regular_file(portal.spec.image)) {
        lua_pushstring(L, ("portal: image '" + portal.spec.image + "' is not a file").c_str());
        return false;
    }
    if (portal.command.empty()) {
        lua_pushstring(L, "portal: it needs a command");
        return false;
    }

    lua_getfield(L, 2, "front");
    const bool FRONT = lua_toboolean(L, -1);
    lua_pop(L, 1);
    const auto& CAM = g_scene.camera();
    if (FRONT) {
        const Vec3 FEET = playerCenter() - Vec3{0.f, Camera::kBodyHeight * 0.5f, 0.f};
        portal.spec.base = FEET + CAM.flatForward() * kPortalFront;
        portal.spec.yaw  = CAM.yaw + std::numbers::pi_v<float>; // facing the player
    } else {
        lua_getfield(L, 2, "at");
        const bool OK = lua_istable(L, -1);
        float xyz[3] = {0.f, 0.f, 0.f};
        for (int k = 0; OK && k < 3; ++k) {
            lua_rawgeti(L, -1, k + 1);
            xyz[k] = static_cast<float>(lua_tonumber(L, -1));
            lua_pop(L, 1);
        }
        lua_pop(L, 1);
        if (!OK) {
            lua_pushstring(L, "portal: it needs at = { x, y, z } or front = true");
            return false;
        }
        portal.spec.base = Vec3{xyz[0], xyz[1], xyz[2]};
        const Vec3 TO = g_playerSpawn - portal.spec.base;
        portal.spec.yaw = NUMBER("yaw", std::atan2(TO.x, -TO.z) * 180.0f / std::numbers::pi_v<float>) *
            std::numbers::pi_v<float> / 180.0f;
    }

    if (IT != g_portals.end())
        *IT = std::move(portal);
    else
        g_portals.push_back(std::move(portal));
    syncPortals();
    damageCurrentMonitor();
    return true;
}

// hl.plugin.hypr3d.object(name, { path = "x.glb", scale = 1, flat = false,
// physics = true }) -- an object put into the room at runtime, the Reddit
// finds' "posters, furniture, a .glb dragged in" (twenty people): 1.2 m in
// front of the player at chest height, it falls to the floor and is
// grabbed, carried and thrown with Super+LMB like the config's dynamic
// objects. physics = false: it stays in the air where it is put and where it
// is carried -- a picture stood up on the floor fell flat (measured).
// object(name) takes it out. The scene table of the next config() that has
// one replaces it -- a new world is a new room. Kept apart from the config's
// objects by a "~" before the name: the list is sorted by name and every
// index-keyed structure follows it, so runtime objects sort after them.
static constexpr float kObjectFront = 1.2f;

static bool objectLua(lua_State* L) {
    if (lua_type(L, 1) != LUA_TSTRING) {
        lua_pushstring(L, "object: the first argument is its name");
        return false;
    }
    const std::string NAME = std::string("~") + lua_tostring(L, 1);
    const auto ERASE = [&](std::vector<SSceneObjectCfg>& list) {
        return std::erase_if(list, [&](const SSceneObjectCfg& o) { return o.name == NAME; });
    };
    if (lua_isnoneornil(L, 2)) {
        ERASE(g_sceneLuaState);
        if (!ERASE(g_sceneObjects)) {
            lua_pushstring(L, ("object: there is no object '" + NAME.substr(1) + "'").c_str());
            return false;
        }
        damageCurrentMonitor();
        return true;
    }
    if (!lua_istable(L, 2)) {
        lua_pushstring(L, "object: the second argument is a table, or nothing to remove it");
        return false;
    }
    SSceneObjectCfg OBJ;
    OBJ.name = NAME;
    lua_getfield(L, 2, "path");
    OBJ.path = lua_type(L, -1) == LUA_TSTRING ? lua_tostring(L, -1) : "";
    lua_pop(L, 1);
    if (OBJ.path.empty() || !std::filesystem::is_regular_file(OBJ.path)) {
        lua_pushstring(L, ("object: path '" + OBJ.path + "' is not a file").c_str());
        return false;
    }
    lua_getfield(L, 2, "scale");
    const float SCALE = lua_isnumber(L, -1) ? std::clamp(static_cast<float>(lua_tonumber(L, -1)), 0.01f, 100.0f) : 1.0f;
    lua_pop(L, 1);
    lua_getfield(L, 2, "flat");
    OBJ.flat = lua_toboolean(L, -1);
    lua_pop(L, 1);
    lua_getfield(L, 2, "physics");
    OBJ.physics = lua_isnil(L, -1) || lua_toboolean(L, -1);
    lua_pop(L, 1);
    OBJ.scale     = Vec3{SCALE, SCALE, SCALE};
    OBJ.collision = true;
    OBJ.dynamic   = true;
    OBJ.center    = CMapModel::ECenter::Logical;
    const auto& CAM = g_scene.camera();
    OBJ.position = playerCenter() + CAM.flatForward() * kObjectFront + Vec3{0.f, 0.3f, 0.f};
    OBJ.rotationDeg = Vec3{0.f, -CAM.yaw * 180.0f / std::numbers::pi_v<float>, 0.f}; // its front to the player

    for (auto* list : {&g_sceneObjects, &g_sceneLuaState}) {
        ERASE(*list);
        list->push_back(OBJ);
        std::sort(list->begin(), list->end(),
                  [](const SSceneObjectCfg& A, const SSceneObjectCfg& B) { return A.name < B.name; });
    }
    damageCurrentMonitor();
    return true;
}

static int luaObject(lua_State* L) {
    if (!objectLua(L))
        return lua_error(L);
    return 0;
}

static int luaPortal(lua_State* L) {
    if (!portalLua(L))
        return lua_error(L);
    return 0;
}

static int luaCompanion(lua_State* L) {
    if (!companionLua(L))
        return lua_error(L);
    return 0;
}

static int luaFocusLock(lua_State* L) {
    // hl.plugin.hypr3d.focus_lock()        lock, or release when locked
    // hl.plugin.hypr3d.focus_lock(true)    the window with the keyboard keeps it
    // hl.plugin.hypr3d.focus_lock(false)   the keyboard follows the crosshair
    setFocusLock(lua_isboolean(L, 1) ? lua_toboolean(L, 1) : g_focusLockId == 0);
    return 0;
}

static int luaToggle(lua_State* L) {
    if (sessionLocked() && !g_active)
        return luaL_error(L, "hypr3d: the session is locked");
    toggle3D();
    return 0;
}

static int luaOpen(lua_State* L) {
    if (sessionLocked())
        return luaL_error(L, "hypr3d: the session is locked");
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

static SDispatchResult dispatchReset(std::string) {
    resetRoom();
    return {};
}

APICALL EXPORT std::string PLUGIN_API_VERSION() {
    return HYPRLAND_API_VERSION;
}

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE handle) {
    joltInit();

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

    if (!HyprlandAPI::addDispatcherV2(PHANDLE, "hypr3d:reset", dispatchReset))
        throw std::runtime_error("[hypr3d] failed to register reset dispatcher");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "toggle", luaToggle))
        throw std::runtime_error("[hypr3d] failed to register Lua toggle");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "open", luaOpen))
        throw std::runtime_error("[hypr3d] failed to register Lua open");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "close", luaClose))
        throw std::runtime_error("[hypr3d] failed to register Lua close");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "portal", luaPortal))
        throw std::runtime_error("[hypr3d] failed to register Lua portal");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "object", luaObject))
        throw std::runtime_error("[hypr3d] failed to register Lua object");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "companion", luaCompanion))
        throw std::runtime_error("[hypr3d] failed to register Lua companion");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "reset", luaReset))
        throw std::runtime_error("[hypr3d] failed to register Lua reset");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "config", luaConfig))
        throw std::runtime_error("[hypr3d] failed to register Lua config");

    if (!HyprlandAPI::addLuaFunction(PHANDLE, "hypr3d", "focus_lock", luaFocusLock))
        throw std::runtime_error("[hypr3d] failed to register Lua focus_lock");

    // Every handler is wrapped: an exception must NEVER escape into
    // Hyprland (std::terminate there kills the whole compositor). The
    // exception text lands in the status dump instead.
    static auto renderPre =
        Event::bus()->m_events.render.pre.listen(
            [](PHLMONITOR mon) {
                try {
                    onRenderPre(mon);
                } catch (const std::exception& e) {
                    dumpErrorNow(std::string{"renderPre: "} + e.what());
                } catch (...) {
                    dumpErrorNow("renderPre: unknown exception");
                }
            }
        );

    static auto renderStage =
        Event::bus()->m_events.render.stage.listen(
            [](eRenderStage stage) {
                try {
                    onRenderStage(stage);
                } catch (const std::exception& e) {
                    dumpErrorNow(std::string{"renderStage: "} + e.what());
                } catch (...) {
                    dumpErrorNow("renderStage: unknown exception");
                }
            }
        );

    static auto mouseMove =
        Event::bus()->m_events.input.mouse.move.listen(
            [](Vector2D pos, Event::SCallbackInfo& info) {
                try {
                    onMouseMove(pos, info);
                } catch (const std::exception& e) {
                    dumpErrorNow(std::string{"mouseMove: "} + e.what());
                } catch (...) {
                    dumpErrorNow("mouseMove: unknown exception");
                }
            }
        );

    static auto mouseButton =
        Event::bus()->m_events.input.mouse.button.listen(
            [](IPointer::SButtonEvent event, Event::SCallbackInfo& info) {
                try {
                    onMouseButton(event, info);
                } catch (const std::exception& e) {
                    dumpErrorNow(std::string{"mouseButton: "} + e.what());
                } catch (...) {
                    dumpErrorNow("mouseButton: unknown exception");
                }
            }
        );

    static auto mouseAxis =
        Event::bus()->m_events.input.mouse.axis.listen(
            [](IPointer::SAxisEvent event, Event::SCallbackInfo& info) {
                try {
                    onMouseAxis(event, info);
                } catch (const std::exception& e) {
                    dumpErrorNow(std::string{"mouseAxis: "} + e.what());
                } catch (...) {
                    dumpErrorNow("mouseAxis: unknown exception");
                }
            }
        );

    static auto keyboardKey =
        Event::bus()->m_events.input.keyboard.key.listen(
            [](IKeyboard::SKeyEvent event, Event::SCallbackInfo& info) {
                try {
                    onKeyboardKey(event, info);
                } catch (const std::exception& e) {
                    dumpErrorNow(std::string{"keyboardKey: "} + e.what());
                } catch (...) {
                    dumpErrorNow("keyboardKey: unknown exception");
                }
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
        "0.5.0"
    };
}

APICALL EXPORT void PLUGIN_EXIT() {
    // Hyprland keeps the last frame's pass elements until the next
    // beginRender() clears them -- ours included. After dlclose() that clear()
    // ran the destructor of a CHypr3DPassElement whose code was gone: unloading
    // while the 3D view was on crashed Hyprland (SIGSEGV in
    // IHyprRenderer::beginRender, measured on 0.56.2). `hyprctl plugin unload`
    // calls this from the event loop, between frames, so the element is done.
    if (g_pHyprRenderer)
        g_pHyprRenderer->m_renderPass.removeAllOfType("Hypr3D");

    if (g_deactivateLater && g_pEventLoopManager)
        g_pEventLoopManager->removeDoLater(g_deactivateLater);
    g_deactivateLater = 0;

    // A bridge must not take a room that is gone for open.
    g_transitionTarget = 0.0f;
    syncRoomState(true);

    joltShutdown();
    headClose();

    g_active = false;
    stopFramePump();
    g_transition = 0.0f;
    g_transitionTarget = 0.0f;

    if (Compat::setCursorHidden(false) && g_pHyprRenderer)
        g_pHyprRenderer->setCursorFromName("default", true);

    clearAimFocus();

    g_world.clear();
    g_renderWindows.clear();

    // Defer the snapshot release to the event loop: this runs INSIDE
    // Hyprland's render pass, and destroying framebuffers/textures here
    // breaks the frame that is still being composed (windows left
    // transparent until their next repaint).
    g_captureReleasePending = true;

    // Put the windows back under the layout before the plugin goes away.
    unghostWindows();

    // The FS fade channel: on FS-on Hyprland dims EVERY other workspace
    // window to alpha 0 (WINDOW_ALPHA_FULLSCREEN). If our transition raced
    // the fade, windows can be left semi-transparent after the 3D exit.
    // Reset the channel for the active workspace unconditionally.
    if (const auto MON = targetMonitor(); MON && MON->m_activeWorkspace) {
        for (auto const& W3 : Desktop::windowState()->windows()) {
            if (W3 && W3->m_workspace == MON->m_activeWorkspace &&
                !W3->m_pinned)
                *W3->alpha(Desktop::View::WINDOW_ALPHA_FULLSCREEN) = 1.F;
        }
    }

    // The ghosted windows were skipped by Hyprland's main render during 3D;
    // force a full monitor repaint so every window is re-composited from its
    // own buffer immediately (static clients would otherwise stay invisible
    // until their first repaint).
    if (g_pHyprRenderer && g_monitor)
        g_pHyprRenderer->damageMonitor(g_monitor);
    damageAllMonitors();

    Compat::setPointerCapture(false);
    Compat::removePointerHook();

    g_monitor = nullptr;

    if (Render::GL::g_pHyprOpenGL) {
        Render::GL::g_pHyprOpenGL->makeEGLCurrent();
        g_scene.shutdown();
    }
}

} // namespace H3D
