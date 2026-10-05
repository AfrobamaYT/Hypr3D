#include "HyprlandCompat/WindowCapture.hpp"

#include "HyprlandCompat/WindowsCompat.hpp"

#include <hyprland/src/desktop/view/Window.hpp>
#include <hyprland/src/desktop/view/LayerSurface.hpp>
#include <hyprland/src/protocols/LayerShell.hpp>
#include <hyprland/src/protocols/core/Compositor.hpp>
#include <hyprland/src/protocols/types/Buffer.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/render/OpenGL.hpp>
#include <hyprland/src/render/gl/GLFramebuffer.hpp>

#include <GLES3/gl32.h>

#include <algorithm>
#include <cmath>

namespace H3D::Compat {

namespace {

// Raw identity of a surface's last-committed buffer. Taking the pointer does
// NOT keep the buffer alive (no reference is held), so the client's buffer
// lifecycle is untouched.
std::uintptr_t bufferIdentity(const WP<CWLSurfaceResource>& surface) {
    if (!surface || !surface->m_current.buffer.m_buffer)
        return 0;

    return reinterpret_cast<std::uintptr_t>(surface->m_current.buffer.m_buffer.get());
}

bool boxEquals(const CBox& a, const CBox& b) {
    return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h;
}

void destroyTexture(unsigned int& tex) {
    if (!tex)
        return;

    glDeleteTextures(1, &tex);
    tex = 0;
}

GLuint createBlankTexture(int width, int height) {
    GLuint tex = 0;
    glGenTextures(1, &tex);

    // Zero-filled, not nullptr: glTexImage2D with null data leaves the
    // contents UNDEFINED, and any tile that fails to capture would then show
    // garbage instead of a clean black patch until the next frame retries.
    std::vector<unsigned char> blank(static_cast<size_t>(width) * height * 4, 0);

    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
        blank.data());

    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

} // namespace

bool CWindowCapture::makeSnapshot(const PHLWINDOW& window, const PHLMONITOR& monitor, bool force) {
    if (!window || !monitor)
        return false;

    if (Render::GL::g_pHyprOpenGL)
        Render::GL::g_pHyprOpenGL->makeEGLCurrent();

    CBox     fullBox{};
    Vector2D surfOffset{}, surfSize{};
    if (!decoratedSurfaceBox(window, monitor, fullBox, surfOffset, surfSize))
        return false;

    const Vector2D MONLOGI = monitor->m_size;

    // Unchanged content and geometry: keep the previous snapshot. The nested
    // snapshot pass is by far the heaviest per-frame cost (a full fake render
    // plus a monitor-sized framebuffer allocation), and skipping it for idle
    // windows is what keeps the room cheap while a screen capture is running.
    const auto ID = reinterpret_cast<std::uintptr_t>(window.get());

    if (!force) {
        if (auto IT = m_snapshots.find(ID); IT != m_snapshots.end() &&
            IT->second.lastBuffer == bufferIdentity(window->resource()) &&
            boxEquals(IT->second.fullBox, fullBox))
            return true;
    }

    // makeSnapshotFB renders into a monitor-sized framebuffer and clips
    // everything outside the viewport. A window sticking past the screen edge
    // is pulled fully into view by warping the workspace render offset for
    // the duration of its pass. A window LARGER than the monitor can never
    // fit at all -- stretching the captured slice over the full quad
    // distorted the content -- so it is captured in monitor-sized tiles and
    // composited into one window-sized texture at native scale, which keeps
    // the true aspect ratio at any size. The 2D desktop underneath stays
    // hidden behind the 3D scene the whole time.

    if (fullBox.w > MONLOGI.x || fullBox.h > MONLOGI.y)
        return makeTiledSnapshot(window, monitor, fullBox, surfOffset, surfSize, ID, force);

    Vector2D shift;
    const auto WORKSPACE = window->m_workspace;

    if (WORKSPACE) {
        const Vector2D MONPOS = monitor->m_position;
        const double   GX     = MONPOS.x + fullBox.x;
        const double   GY     = MONPOS.y + fullBox.y;

        const double LEFT   = MONPOS.x - GX;
        const double RIGHT  = (GX + fullBox.w) - (MONPOS.x + MONLOGI.x);
        const double TOP    = MONPOS.y - GY;
        const double BOTTOM = (GY + fullBox.h) - (MONPOS.y + MONLOGI.y);

        if (LEFT > 0.0)
            shift.x = LEFT;
        else if (RIGHT > 0.0)
            shift.x = -RIGHT;

        if (TOP > 0.0)
            shift.y = TOP;
        else if (BOTTOM > 0.0)
            shift.y = -BOTTOM;
    }

    const bool     SHIFTED  = shift.x != 0.0 || shift.y != 0.0;
    const Vector2D ORIGINAL = WORKSPACE ? WORKSPACE->m_renderOffset->value() : Vector2D{};

    if (SHIFTED)
        WORKSPACE->m_renderOffset->setValueAndWarp(shift);

    // Renders the window (content + decorations) into a fresh plugin-ownable
    // framebuffer. Requires the window to currently pass shouldRenderWindow.
    auto fb = g_pHyprRenderer->makeSnapshotFB(window);

    if (SHIFTED)
        WORKSPACE->m_renderOffset->setValueAndWarp(ORIGINAL);

    if (!fb)
        return false;

    const auto TEXTURE = fb->getTexture();

    SSnapshot snapshot;
    snapshot.fb     = fb;
    snapshot.texID  = TEXTURE ? TEXTURE->m_texID : 0;
    snapshot.width  = TEXTURE ? static_cast<int>(TEXTURE->m_size.x) : 0;
    snapshot.height = TEXTURE ? static_cast<int>(TEXTURE->m_size.y) : 0;

    if (snapshot.texID == 0 || snapshot.width <= 0 || snapshot.height <= 0) {
        snapshot.fb = nullptr;
        return false;
    }

    // Record the geometry the snapshot was rendered at. The live box may be
    // written later in the same frame (the resize path runs after render.pre),
    // and the UV subrect must stay aligned with the captured pixels, not with
    // the live geometry.
    snapshot.fullBox = fullBox;
    snapshot.sampledBox = CBox{
        fullBox.x + static_cast<int>(shift.x),
        fullBox.y + static_cast<int>(shift.y),
        fullBox.w,
        fullBox.h,
    };
    snapshot.surfaceOffset = surfOffset;
    snapshot.surfaceSize   = surfSize;
    snapshot.texSpan       = MONLOGI;
    snapshot.lastBuffer    = bufferIdentity(window->resource());

    // Drop the previous snapshot first: the SP releases the framebuffer's GL
    // objects on destruction, and a leftover composite texture from a tiled
    // snapshot has to be deleted explicitly.
    if (auto IT = m_snapshots.find(ID); IT != m_snapshots.end()) {
        destroySnapshotGL(IT->second);
        m_snapshots.erase(IT);
    }
    m_snapshots.emplace(ID, std::move(snapshot));

    return true;
}

// A window larger than the monitor cannot fit into a single monitor-sized
// snapshot. Capture it as a grid of monitor-sized tiles -- each pass has the
// workspace render offset warped so its tile starts at the viewport origin --
// and copy every tile into one window-sized texture. The composite keeps the
// window's true aspect ratio at native pixel density: no distortion at all.
bool CWindowCapture::makeTiledSnapshot(
    const PHLWINDOW&  window,
    const PHLMONITOR& monitor,
    const CBox&       fullBox,
    const Vector2D&   surfOffset,
    const Vector2D&   surfSize,
    std::uintptr_t    id,
    bool              force
) {
    const Vector2D MONLOGI = monitor->m_size;
    const Vector2D MONPIX  = monitor->m_pixelSize;

    const double SX = MONPIX.x / MONLOGI.x; // logical -> texture px
    const double SY = MONPIX.y / MONLOGI.y;

    GLint maxTex = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);

    const int TEXW = std::min<GLint>(maxTex, static_cast<GLint>(std::lround(fullBox.w * SX)));
    const int TEXH = std::min<GLint>(maxTex, static_cast<GLint>(std::lround(fullBox.h * SY)));

    if (TEXW <= 0 || TEXH <= 0)
        return false;

    const auto IT = m_snapshots.find(id);

    // Unchanged content and geometry: keep the previous composite. The tile
    // grid is the most expensive capture path there is.
    if (!force && IT != m_snapshots.end() &&
        IT->second.lastBuffer == bufferIdentity(window->resource()) &&
        boxEquals(IT->second.fullBox, fullBox))
        return true;

    // Reuse the composite texture while the box size is unchanged.
    GLuint       bigTex = 0;
    if (IT != m_snapshots.end() && IT->second.bigTex &&
        static_cast<int>(IT->second.texSpan.x) == static_cast<int>(fullBox.w) &&
        static_cast<int>(IT->second.texSpan.y) == static_cast<int>(fullBox.h)) {
        bigTex = IT->second.bigTex;
    } else {
        if (IT != m_snapshots.end() && IT->second.bigTex)
            destroyTexture(IT->second.bigTex);

        bigTex = createBlankTexture(TEXW, TEXH);

        if (!bigTex)
            return false;
    }

    const auto WORKSPACE = window->m_workspace;
    const Vector2D ORIGINAL =
        WORKSPACE ? WORKSPACE->m_renderOffset->value() : Vector2D{};

    const int COLS = static_cast<int>(std::ceil(fullBox.w / MONLOGI.x));
    const int ROWS = static_cast<int>(std::ceil(fullBox.h / MONLOGI.y));

    bool anyTile = false;

    for (int j = 0; j < ROWS; ++j) {
        for (int i = 0; i < COLS; ++i) {
            const float TILE_X = i * MONLOGI.x;
            const float TILE_Y = j * MONLOGI.y;
            const float TILE_W = std::min<float>(MONLOGI.x, fullBox.w - TILE_X);
            const float TILE_H = std::min<float>(MONLOGI.y, fullBox.h - TILE_Y);

            // Warp the workspace offset so THIS tile's top-left corner sits
            // at the viewport origin; the rest of the window clips away.
            const Vector2D shift{
                -fullBox.x - TILE_X,
                -fullBox.y - TILE_Y,
            };

            if (WORKSPACE)
                WORKSPACE->m_renderOffset->setValueAndWarp(shift);

            auto fb = g_pHyprRenderer->makeSnapshotFB(window);

            if (WORKSPACE)
                WORKSPACE->m_renderOffset->setValueAndWarp(ORIGINAL);

            if (!fb)
                continue; // hole in the composite; next frame retries

            auto* GLFB = dynamic_cast<Render::GL::CGLFramebuffer*>(fb.get());

            if (!GLFB)
                continue;

            const int SRCW = static_cast<int>(std::lround(TILE_W * SX));
            const int SRCH = static_cast<int>(std::lround(TILE_H * SY));

            // FB row 0 holds the tile's logical top row; copying it to dest
            // row TILE_Y*SY keeps the composite's row 0 at the window's
            // logical top -- the same top-down convention the monitor-sized
            // snapshots are sampled with.
            glBindFramebuffer(GL_READ_FRAMEBUFFER, GLFB->getFBID());
            glBindTexture(GL_TEXTURE_2D, bigTex);
            glCopyTexSubImage2D(
                GL_TEXTURE_2D, 0,
                static_cast<int>(std::lround(TILE_X * SX)),
                static_cast<int>(std::lround(TILE_Y * SY)),
                0, 0, SRCW, SRCH);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

            anyTile = true;
        }
    }

    if (!anyTile)
        return false;

    // Drop a monitor-sized fb left over from a previous non-tiled snapshot;
    // the composite texture replaces it wholesale.
    SSnapshot snapshot;
    snapshot.fb     = nullptr;
    snapshot.texID  = 0;
    snapshot.bigTex = bigTex;
    snapshot.width  = TEXW;
    snapshot.height = TEXH;

    snapshot.fullBox       = fullBox;
    snapshot.sampledBox    = CBox{0, 0, fullBox.w, fullBox.h};
    snapshot.surfaceOffset = surfOffset;
    snapshot.surfaceSize   = surfSize;
    snapshot.texSpan       = Vector2D{fullBox.w, fullBox.h};
    snapshot.lastBuffer    = bufferIdentity(window->resource());

    m_snapshots.erase(id);
    m_snapshots.emplace(id, std::move(snapshot));

    return true;
}

// Layer-shell surfaces render at their monitor-local box with no
// decorations: the snapshot is monitor-sized with the layer already in
// place, so the UV math matches the window path with a zero surface offset.
// No off-screen pull here -- layers are anchored to the monitor by the shell.
bool CWindowCapture::makeSnapshotLayer(const PHLLS& layer, const PHLMONITOR& monitor, bool force) {
    if (!layer || !monitor)
        return false;

    if (Render::GL::g_pHyprOpenGL)
        Render::GL::g_pHyprOpenGL->makeEGLCurrent();

    const auto BOXOPT = layer->surfaceLogicalBox();

    if (!BOXOPT || BOXOPT->w <= 0 || BOXOPT->h <= 0)
        return false;

    const auto ID = reinterpret_cast<std::uintptr_t>(layer.get());

    const auto SURFACE =
        layer->m_layerSurface ? layer->m_layerSurface->m_surface.lock() : nullptr;

    if (!force) {
        if (auto IT = m_snapshots.find(ID); IT != m_snapshots.end() &&
            IT->second.lastBuffer == bufferIdentity(SURFACE) &&
            boxEquals(IT->second.fullBox, *BOXOPT))
            return true;
    }

    auto fb = g_pHyprRenderer->makeSnapshotFB(layer);

    if (!fb)
        return false;

    const auto TEXTURE = fb->getTexture();

    SSnapshot snapshot;
    snapshot.fb     = fb;
    snapshot.texID  = TEXTURE ? TEXTURE->m_texID : 0;
    snapshot.width  = TEXTURE ? static_cast<int>(TEXTURE->m_size.x) : 0;
    snapshot.height = TEXTURE ? static_cast<int>(TEXTURE->m_size.y) : 0;

    if (snapshot.texID == 0 || snapshot.width <= 0 || snapshot.height <= 0) {
        snapshot.fb = nullptr;
        return false;
    }

    snapshot.fullBox       = *BOXOPT;
    snapshot.sampledBox    = *BOXOPT;
    snapshot.surfaceOffset = Vector2D{0, 0};
    snapshot.surfaceSize   = Vector2D{BOXOPT->w, BOXOPT->h};
    snapshot.texSpan       = monitor->m_size;

    // Alpha mask for picking, refreshed every 32nd snapshot (or when the box
    // size changed): a downsampled readback of the layer's box region.
    // Invisible overlays then let the crosshair fall through their empty
    // pixels instead of swallowing input for the window underneath.
    ++snapshot.maskAge;

    const double SX = monitor->m_pixelSize.x / monitor->m_size.x;
    const double SY = monitor->m_pixelSize.y / monitor->m_size.y;

    const int PW = static_cast<int>(std::lround(BOXOPT->w * SX));
    const int PH = static_cast<int>(std::lround(BOXOPT->h * SY));

    const bool SIZE_CHANGED =
        snapshot.alphaValid &&
        (snapshot.alphaW * 4 != PW || snapshot.alphaH * 4 != PH);

    if (!snapshot.alphaValid || SIZE_CHANGED || snapshot.maskAge % 32 == 1) {
        auto* GLFB = dynamic_cast<Render::GL::CGLFramebuffer*>(fb.get());

        // The layer sits at its box position inside the monitor-sized
        // snapshot; read exactly that region, clamped to the framebuffer.
        const int PX = std::clamp(static_cast<int>(std::lround(BOXOPT->x * SX)), 0,
            std::max(0, snapshot.width - 1));
        const int PY = std::clamp(static_cast<int>(std::lround(BOXOPT->y * SY)), 0,
            std::max(0, snapshot.height - 1));
        const int RW = std::min(PW, snapshot.width - PX);
        const int RH = std::min(PH, snapshot.height - PY);

        if (GLFB && RW > 0 && RH > 0) {
            std::vector<unsigned char> rgba(static_cast<size_t>(RW) * RH * 4);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, GLFB->getFBID());
            glReadPixels(PX, PY, RW, RH, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

            const int MW = std::max(1, RW / 4);
            const int MH = std::max(1, RH / 4);

            std::vector<unsigned char> mask(static_cast<size_t>(MW) * MH, 255);

            for (int j = 0; j < MH; ++j) {
                const int SRCY = std::min(RH - 1, j * 4);

                for (int i = 0; i < MW; ++i) {
                    const int SRCX = std::min(RW - 1, i * 4);
                    mask[static_cast<size_t>(j) * MW + i] =
                        rgba[(static_cast<size_t>(SRCY) * RW + SRCX) * 4 + 3];
                }
            }

            snapshot.alphaMask  = std::move(mask);
            snapshot.alphaW     = MW;
            snapshot.alphaH     = MH;
            snapshot.alphaValid = true;

            // The same mask doubles as the depth-slab silhouette source:
            // bars and panels get contour-hugging walls for free.
            snapshot.outlines =
                std::make_shared<const std::vector<H3D::SOutlineLoop>>(
                    H3D::traceOutlines(snapshot.alphaMask.data(), MW, MH, 64,
                                       1.5f, 4, 128));
        }
    }

    snapshot.lastBuffer = bufferIdentity(SURFACE);

    m_snapshots.erase(ID);
    m_snapshots.emplace(ID, std::move(snapshot));

    return true;
}

bool CWindowCapture::has(std::uintptr_t id) const {
    return m_snapshots.contains(id);
}

const CWindowCapture::SSnapshot* CWindowCapture::get(std::uintptr_t id) const {
    const auto IT = m_snapshots.find(id);

    return IT == m_snapshots.end() ? nullptr : &IT->second;
}

void CWindowCapture::destroySnapshotGL(SSnapshot& snapshot) {
    if (snapshot.bigTex && Render::GL::g_pHyprOpenGL)
        Render::GL::g_pHyprOpenGL->makeEGLCurrent();

    destroyTexture(snapshot.bigTex);

    // The monitor-sized fb frees its GL objects through the SP.
    snapshot.fb    = nullptr;
    snapshot.texID = 0;
}

void CWindowCapture::release(std::uintptr_t id) {
    if (auto IT = m_snapshots.find(id); IT != m_snapshots.end()) {
        destroySnapshotGL(IT->second);
        m_snapshots.erase(IT);
    }
}

void CWindowCapture::releaseAll() {
    for (auto& [id, snapshot] : m_snapshots)
        destroySnapshotGL(snapshot);

    m_snapshots.clear();
}

void CWindowCapture::retainOnly(const std::vector<std::uintptr_t>& keep) {
    for (auto IT = m_snapshots.begin(); IT != m_snapshots.end();) {
        const bool FOUND =
            std::find(keep.begin(), keep.end(), IT->first) != keep.end();

        if (FOUND)
            ++IT;
        else {
            destroySnapshotGL(IT->second);
            IT = m_snapshots.erase(IT);
        }
    }
}

void CWindowCapture::shutdownGL() {
    releaseAll();
}

} // namespace H3D::Compat
