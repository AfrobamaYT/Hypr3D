#include "HyprlandCompat/WindowsCompat.hpp"

#include <hyprland/src/desktop/state/FocusState.hpp>
#include <hyprland/src/desktop/state/WindowState.hpp>
#include <hyprland/src/desktop/view/Window.hpp>
#include <hyprland/src/desktop/view/Popup.hpp>
#include <hyprland/src/desktop/view/LayerSurface.hpp>
#include <hyprland/src/desktop/Workspace.hpp>
#include <hyprland/src/layout/target/Target.hpp>
#include <hyprland/src/config/shared/actions/ConfigActions.hpp>
#include <hyprland/src/managers/SeatManager.hpp>
#include <hyprland/src/protocols/LayerShell.hpp>
#include <hyprland/src/protocols/PointerConstraints.hpp>
#include <hyprland/src/desktop/view/WLSurface.hpp>
// CLayerShellResource's m_surface (CWLSurfaceResource) needs the full type
// for .lock()->m_current access in deliver* paths.
#include <hyprland/src/protocols/types/SurfaceState.hpp>
#include <hyprland/src/state/MonitorState.hpp>
#include <hyprland/src/render/Renderer.hpp>

#include <hyprutils/utils/ScopeGuard.hpp>
#include <cmath>

namespace H3D::Compat {

std::vector<SWindowInfo> enumerateEligibleWindows(const PHLMONITOR& monitor) {
    std::vector<SWindowInfo> out;

    if (!monitor || !Desktop::windowState())
        return out;

    for (const auto& window : Desktop::windowState()->windows()) {
        if (!window || !window->m_isMapped || window->isHidden())
            continue;

        const auto WORKSPACE = window->m_workspace;

        if (!WORKSPACE)
            continue;

        // The window must live on a workspace attached to this monitor that is
        // actually being rendered (the snapshot path requires it).
        if (WORKSPACE->m_monitor != monitor)
            continue;

        if (!WORKSPACE->isVisible() && !window->m_pinned && !WORKSPACE->m_forceRendering)
            continue;

        SWindowInfo info;
        info.id       = reinterpret_cast<std::uintptr_t>(window.get());
        info.window   = window;
        info.floating = window->m_isFloating;
        info.monitor  = monitor;

        if (!decoratedSurfaceBox(
                window,
                monitor,
                info.monitorLocalBox,
                info.surfaceOffset,
                info.surfaceSize))
            continue;

        out.push_back(std::move(info));
    }

    // Layer-shell surfaces (panels, bars, quickshell PanelWindow) live
    // outside the window list; show them in the room as regular entities.
    // The background layer is skipped -- that is where wallpapers live, and a
    // full-screen wallpaper would swallow the whole room. Their boxes come in
    // layout coordinates and are made monitor-local below; decoration-free.
    constexpr uint32_t LAYER_BACKGROUND = 0;

    for (const auto& LAYERLIST : monitor->m_layerSurfaceLayers) {
        for (const auto& LSREF : LAYERLIST) {
            const auto LS = LSREF.lock();

            if (!LS || !LS->visible() || LS->m_layer == LAYER_BACKGROUND)
                continue;

            const auto BOXOPT = LS->surfaceLogicalBox();

            if (!BOXOPT || BOXOPT->w <= 0 || BOXOPT->h <= 0)
                continue;

            SWindowInfo info;
            info.id      = reinterpret_cast<std::uintptr_t>(LS.get());
            info.layer   = LS;
            info.isLayer = true;
            info.monitor = monitor;

            // Layout coordinates (arrangeLayerArray starts from the monitor's
            // position) made monitor-local like a window's box.
            info.monitorLocalBox = *BOXOPT;
            info.monitorLocalBox.x -= monitor->m_position.x;
            info.monitorLocalBox.y -= monitor->m_position.y;
            info.surfaceOffset   = Vector2D{0, 0};
            info.surfaceSize     = Vector2D{BOXOPT->w, BOXOPT->h};

            out.push_back(std::move(info));
        }
    }

    return out;
}

std::vector<CBox> windowPopupBoxes(const PHLWINDOW& window, const PHLMONITOR& monitor) {
    std::vector<CBox> boxes;
    if (!window || !monitor || !window->m_popupHead) return boxes;
    window->m_popupHead->breadthfirst([&](SP<Desktop::View::CPopup> popup, void*) {
        if (!popup->m_mapped || popup->inert() || !popup->resource()) return;
        if (auto box = popup->surfaceLogicalBox()) {
            box->x -= monitor->m_position.x;
            box->y -= monitor->m_position.y;
            if (box->w > 0 && box->h > 0) boxes.push_back(*box);
        }
    }, nullptr);
    return boxes;
}

bool decoratedSurfaceBox(
    const PHLWINDOW&  window,
    const PHLMONITOR& monitor,
    CBox&             fullBox,
    Vector2D&         surfOffset,
    Vector2D&         surfSize
) {
    if (!window || !monitor)
        return false;

    // The main surface box is the content only; decorations (the border ring,
    // titlebars) extend beyond it and are part of what a snapshot renders, so
    // the 3D quad must span the unified box to show them.
    const auto SURF = window->getWindowMainSurfaceBox();
    const auto FULL = window->getWindowBoxUnified(Desktop::View::FULL_EXTENTS);

    // A dimAround rule makes the unified box cover the whole monitor; clamp
    // each side margin so a rogue rule cannot blow the quad up.
    constexpr double MAX_MARGIN = 150.0;
    const double LEFT   = std::clamp<double>(SURF.x - FULL.x, 0.0, MAX_MARGIN);
    const double TOP    = std::clamp<double>(SURF.y - FULL.y, 0.0, MAX_MARGIN);
    const double RIGHT  = std::clamp<double>((FULL.x + FULL.w) - (SURF.x + SURF.w), 0.0, MAX_MARGIN);
    const double BOTTOM = std::clamp<double>((FULL.y + FULL.h) - (SURF.y + SURF.h), 0.0, MAX_MARGIN);

    const auto MONPOS = monitor->m_position;

    fullBox = CBox{
        SURF.x - LEFT - MONPOS.x,
        SURF.y - TOP - MONPOS.y,
        SURF.w + LEFT + RIGHT,
        SURF.h + TOP + BOTTOM,
    };

    // Real popup surfaces can be larger than decoration margins. Capture
    // their whole tree without letting a dimAround rule expand the window.
    for (const auto& popup : windowPopupBoxes(window, monitor)) {
        const double right = std::max(fullBox.x + fullBox.w, popup.x + popup.w);
        const double bottom = std::max(fullBox.y + fullBox.h, popup.y + popup.h);
        fullBox.x = std::min(fullBox.x, popup.x);
        fullBox.y = std::min(fullBox.y, popup.y);
        fullBox.w = right - fullBox.x;
        fullBox.h = bottom - fullBox.y;
    }

    // Surface-local input coordinates are the hit position minus this offset,
    // so growing the quad to include borders never shifts input.
    surfOffset = Vector2D{SURF.x - MONPOS.x - fullBox.x, SURF.y - MONPOS.y - fullBox.y};
    surfSize   = Vector2D{SURF.w, SURF.h};

    return fullBox.w > 0 && fullBox.h > 0;
}

bool isWindowEligible(const PHLWINDOW& window, const PHLMONITOR& monitor) {
    if (!window || !window->m_isMapped || window->isHidden())
        return false;

    const auto WORKSPACE = window->m_workspace;

    if (!WORKSPACE || !monitor)
        return false;

    if (WORKSPACE->m_monitor != monitor)
        return false;

    if (!WORKSPACE->isVisible() && !window->m_pinned && !WORKSPACE->m_forceRendering)
        return false;

    return true;
}

SWindowLayoutSave saveWindowLayout(const PHLWINDOW& window) {
    SWindowLayoutSave save;

    if (!window || !window->m_target)
        return save;

    save.id          = reinterpret_cast<std::uintptr_t>(window.get());
    save.window      = window;
    save.box         = window->m_target->position();
    save.space       = window->m_target->space();
    save.wasFloating = window->m_target->floating();
    save.fs          = Fullscreen::controller()->getFullscreenModes(window);

    return save;
}

void applyWindowGhost(SWindowLayoutSave& save) {
    if (save.window.expired())
        return;

    const auto WINDOW = save.window.lock();

    if (!WINDOW || !WINDOW->m_target)
        return;

    // The 3D view owns geometry. Force the target floating first, then ghost
    // it out of the layout so a tiled layout cannot overwrite our changes.
    WINDOW->m_target->setFloating(true);
    WINDOW->m_target->setSpaceGhost(save.space);
}

void restoreWindowLayout(SWindowLayoutSave& save) {
    if (save.window.expired())
        return;

    const auto WINDOW = save.window.lock();

    if (!WINDOW || !WINDOW->m_target)
        return;

    // A window that unmapped while the room was open -- hidden by its app,
    // as Steam hides its small windows, or fading out after a close -- is
    // alive without a workspace. Hyprland's assignToSpace moves a window
    // from its workspace and looked at the missing one: SIGSEGV in
    // CWorkspace::isVisible, the owner's Hyprland in safe mode (2026-10-08
    // 23:52, a 147x198 Steam window; reproduced with span/unmap-crash.sh).
    // It only leaves the room's ghost; mapped again, Hyprland lays it out
    // as any window it maps.
    if (!WINDOW->m_workspace) {
        WINDOW->m_target->assignToSpace(nullptr);
        WINDOW->m_target->setFloating(save.wasFloating);
        return;
    }

    // Put the target back into its space while it is still floating. For a
    // window that was floating before 3D, retain the real box produced by the
    // 3D resize. Tiled windows deliberately return to their original layout.
    const CBox RESTORE_BOX = save.wasFloating ? WINDOW->m_target->position() : save.box;

    if (save.space)
        WINDOW->m_target->assignToSpace(save.space);
    else
        WINDOW->m_target->setSpaceGhost(nullptr);

    WINDOW->m_target->setFloating(true);
    WINDOW->m_target->setPositionGlobal(RESTORE_BOX);
    WINDOW->m_target->rememberFloatingSize(Vector2D{RESTORE_BOX.w, RESTORE_BOX.h});

    // A tile goes back the way Hyprland's own settiled takes it: its space
    // re-admits it and lays it out. Flipping the flag alone left it with the
    // box set above -- two tiles side by side touched with no gap (915x1040
    // instead of 908x1038), and a window opened in the room lay over its
    // neighbours' tiles as a "tiled" 960x540 box: the owner's foot and Brave
    // on top of each other on DP-3 (2026-10-08).
    if (!save.wasFloating)
        Config::Actions::floatWindow(Config::Actions::TOGGLE_ACTION_DISABLE, WINDOW);

    // Fullscreen again, the way Hyprland's own fullscreen dispatcher sets
    // it. Left out, a game that was fullscreen came back a floating
    // monitor-sized window over its workspace's tiles: they could not be
    // clicked and new windows tiled under it (the owner's CS2 over Steam,
    // 2026-10-08). The room took the window out of its fullscreen box
    // without telling Hyprland, which still holds it fullscreen: asked for
    // fullscreen again it changed nothing (measured: 2/2 before and after,
    // the box a 458 px tile). So out first, then in.
    if (save.fs.internal != Fullscreen::FSMODE_NONE || save.fs.client != Fullscreen::FSMODE_NONE) {
        Fullscreen::controller()->setFullscreenMode(WINDOW, Fullscreen::FSMODE_NONE, Fullscreen::FSMODE_NONE);
        Fullscreen::controller()->setFullscreenMode(WINDOW, save.fs.internal, save.fs.client);
    }

    g_pHyprRenderer->damageWindow(WINDOW);
}

CBox currentWindowBox(const PHLWINDOW& window) {
    if (!window || !window->m_target)
        return {};

    return window->m_target->position();
}

bool setWindowBox(const PHLWINDOW& window, const CBox& box) {
    if (!window || !window->m_target)
        return false;

    const CBox CURRENT = window->m_target->position();

    const bool unchanged =
        std::fabs(CURRENT.x - box.x) < 0.01 &&
        std::fabs(CURRENT.y - box.y) < 0.01 &&
        std::fabs(CURRENT.w - box.w) < 0.01 &&
        std::fabs(CURRENT.h - box.h) < 0.01;

    if (unchanged)
        return false;

    window->m_target->setFloating(true);
    window->m_target->setPositionGlobal(box);
    window->m_target->rememberFloatingSize(Vector2D{box.w, box.h});

    if (g_pHyprRenderer)
        g_pHyprRenderer->damageWindow(window);

    return true;
}

PHLLS findLayerById(std::uintptr_t id) {
    if (id == 0)
        return nullptr;

    for (const auto& MONITOR : State::monitorState()->monitors()) {
        for (const auto& LAYERLIST : MONITOR->m_layerSurfaceLayers) {
            for (const auto& LSREF : LAYERLIST) {
                const auto LS = LSREF.lock();

                if (LS && reinterpret_cast<std::uintptr_t>(LS.get()) == id)
                    return LS;
            }
        }
    }

    return nullptr;
}

void clearPointerFocus() {
    if (!g_pSeatManager)
        return;

    g_pSeatManager->setPointerFocus(nullptr, {});
}

// The point the room last sent a window, so that it sends motion only when
// that point moved, as a mouse does. The room aims every frame; each position
// sent to an Xwayland window becomes an X raw event in screen coordinates,
// which a game that took the mouse reads as that much motion (GLFW takes raw
// values as deltas): ~1000 px per frame, the spinbot, and on clicking back
// into a game the position sent with the click before its lock arrived
// (measured with XI2 raw events, 2026-10-07).
static WP<CWLSurfaceResource> g_motionSurface;
static Vector2D               g_motionLocal;

// Focuses SURFACE at LOCAL; true when the client is to be told the position:
// on an enter, or when the point moved by wl_fixed's 1/256 or more.
static bool pointerMoved(const SP<CWLSurfaceResource>& surface, const Vector2D& local) {
    const bool ENTER = g_pSeatManager->m_state.pointerFocus != surface;
    g_pSeatManager->setPointerFocus(surface, local);
    if (!ENTER && g_motionSurface == surface && (local - g_motionLocal).size() < 1.0 / 256.0)
        return false;
    g_motionSurface = surface;
    g_motionLocal   = local;
    return true;
}

int pointerConstraintOf(const PHLWINDOW& window) {
    if (!window || !window->resource())
        return 0;
    const auto HL = Desktop::View::CWLSurface::fromResource(window->resource());
    const auto C  = HL ? HL->constraint() : nullptr;
    if (!C || !C->isActive())
        return 0;
    return C->isLocked() ? 2 : 1;
}

// Native child surfaces own their pointer events; passing the coordinates
// to the toplevel instead made visible menus impossible to select.
static std::pair<SP<CWLSurfaceResource>, Vector2D> pointerSurface(
    const PHLWINDOW& window, const Vector2D& local) {
    const auto global = window->getWindowMainSurfaceBox().pos() + local;
    if (window->m_popupHead)
        if (const auto popup = window->m_popupHead->at(global, true))
            if (const auto surface = popup->resource())
                return {surface, global - popup->coordsGlobal()};
    return {window->resource(), local};
}

void deliverMotion(
    const PHLWINDOW& window,
    const Vector2D& localLogical,
    uint32_t timeMs
) {
    if (!window || !g_pSeatManager)
        return;

    // A locked pointer gets no motion events (zwp_locked_pointer_v1: "the
    // wl_pointer objects of the associated seat will not emit any
    // wl_pointer.motion events"), only Hyprland's relative motion. A game
    // that had grabbed the mouse read each absolute position as more motion
    // and spun (measured 2026-10-07).
    const auto SURFACE = window->resource();
    if (!SURFACE)
        return;

    // Hyprland lets every lock go when a pointer device goes away
    // (CInputManager::destroyPointer -> unconstrainMouse) and gives it back
    // only on the next focus change -- which in the room never comes, so the
    // game spun on from there. The focused window gets its lock back here,
    // as a focus change would give it; a oneshot lock died with the
    // deactivation and activate() leaves it so.
    if (Desktop::focusState()->surface() == SURFACE) {
        const auto HL = Desktop::View::CWLSurface::fromResource(SURFACE);
        if (const auto C = HL ? HL->constraint() : nullptr; C && C->isLocked() && !C->isActive())
            C->activate();
    }
    if (pointerConstraintOf(window) == 2) {
        g_pSeatManager->setPointerFocus(SURFACE, localLogical); // keeps it; a no-op when it has it
        return;
    }

    const auto [pointed, local] = pointerSurface(window, localLogical);
    if (!pointerMoved(pointed, local))
        return;

    g_pSeatManager->sendPointerMotion(timeMs, local);
    g_pSeatManager->sendPointerFrame();
}

void deliverClick(
    const PHLWINDOW& window,
    const Vector2D&  localLogical,
    uint32_t         button,
    bool             pressed,
    uint32_t         timeMs
) {
    if (!window)
        return;

    const auto SURFACE = window->resource();

    if (!SURFACE || !g_pSeatManager)
        return;

    // Point the seat at the aimed surface, put the virtual pointer at the hit
    // coordinate, then send the button and a frame. The application therefore
    // sees an ordinary Wayland pointer event even though the physical cursor
    // is captured by the 3D view. A locked pointer gets the button alone: a
    // position with every click was a jump with every shot in a game. So
    // does a pointer already at the point (pointerMoved).
    if (pointerConstraintOf(window) == 2)
        g_pSeatManager->setPointerFocus(SURFACE, localLogical);
    else {
        const auto [pointed, local] = pointerSurface(window, localLogical);
        if (pointerMoved(pointed, local))
            g_pSeatManager->sendPointerMotion(timeMs, local);
    }
    g_pSeatManager->sendPointerButton(
        timeMs,
        button,
        pressed ? WL_POINTER_BUTTON_STATE_PRESSED : WL_POINTER_BUTTON_STATE_RELEASED
    );
    g_pSeatManager->sendPointerFrame();
}

void deliverMotion(
    const PHLLS& layer,
    const Vector2D& localLogical,
    uint32_t timeMs
) {
    if (!layer || !g_pSeatManager)
        return;

    const auto SURFACE =
        layer->m_layerSurface ? layer->m_layerSurface->m_surface.lock() : nullptr;

    if (!SURFACE)
        return;

    g_pSeatManager->setPointerFocus(SURFACE, localLogical);
    g_pSeatManager->sendPointerMotion(timeMs, localLogical);
    g_pSeatManager->sendPointerFrame();
}

void deliverClick(
    const PHLLS& layer,
    const Vector2D& localLogical,
    uint32_t button,
    bool pressed,
    uint32_t timeMs
) {
    if (!layer || !g_pSeatManager)
        return;

    const auto SURFACE =
        layer->m_layerSurface ? layer->m_layerSurface->m_surface.lock() : nullptr;

    if (!SURFACE)
        return;

    g_pSeatManager->setPointerFocus(SURFACE, localLogical);
    g_pSeatManager->sendPointerMotion(timeMs, localLogical);
    g_pSeatManager->sendPointerButton(
        timeMs,
        button,
        pressed ? WL_POINTER_BUTTON_STATE_PRESSED : WL_POINTER_BUTTON_STATE_RELEASED
    );
    g_pSeatManager->sendPointerFrame();
}

} // namespace H3D::Compat
