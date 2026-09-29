#pragma once

#include "World/Camera.hpp"
#include "World/Picking.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace H3D {

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

        // Subrect of `texture` to sample: (u0,v0) at the quad's bottom-left in
        // GL convention, (u1,v1) at its top-right. Defaults cover the whole
        // texture so a caller with a tightly-fit texture needs no bookkeeping.
        float u0 = 0.f, v0 = 0.f, u1 = 1.f, v1 = 1.f;

        float alpha = 1.0f;
    };

    GLScene();

    // Draws into `targetFBO` and composites the result over whatever is
    // already there with `alpha`, so the 2D desktop underneath fades out
    // instead of being captured and re-projected -- the workspace itself is
    // never rendered as an object in the scene.
    bool render(unsigned int targetFBO, int width, int height, float alpha,
                float dt, const std::vector<WindowRender>& windows);

    // Raw pointer counts -> camera look. Prefer driving this through
    // InputController so sensitivity policy stays in one testable place.
    void mouseMove(float dx, float dy);

    // Already-converted rotation in radians.
    void rotateView(float yawDelta, float pitchDelta);

    // Background panorama: set a path ("" restores the default void). The
    // image is (re)loaded lazily inside render() when the file changes.
    // Equirectangular mapping around the camera.
    void setPanoramaPath(const std::string& path);

    // Backdrop blur behind transparent window pixels, mirroring Hyprland's
    // decoration:blur config (pushed every frame by main.cpp).
    void setBlurConfig(bool enabled, int size, int passes, float vibrancy);

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
    void drawPanorama(float aspect);
    void refreshPanorama();

    bool ensureBlurTargets(int width, int height);
    bool renderBlur(int width, int height);
    void destroyBlurTargets();
    void drawWindows(const Mat4& vp, const std::vector<WindowRender>& windows,
                     bool frost = false);
    void drawFullscreen(float alpha);
    void drawCrosshair(int width, int height);

  private:
    bool m_initialized = false;

    int m_sceneWidth = 0;
    int m_sceneHeight = 0;

    unsigned int m_sceneFBO = 0;
    unsigned int m_sceneColor = 0;
    unsigned int m_sceneDepth = 0;

    unsigned int m_sceneProgram = 0;
    unsigned int m_blitProgram = 0;
    unsigned int m_panoramaProgram = 0;

    // Unit quad in the XY plane, centred on the origin, facing +Z. Windows.
    unsigned int m_quadVAO = 0;
    unsigned int m_quadVBO = 0;
    int          m_quadVertexCount = 0;

    // Unit quad in the XZ plane. The ground.
    unsigned int m_floorVAO = 0;
    unsigned int m_floorVBO = 0;
    int          m_floorVertexCount = 0;

    unsigned int m_gridVAO = 0;
    unsigned int m_gridVBO = 0;
    int          m_gridVertexCount = 0;

    unsigned int m_fullscreenVAO = 0;
    unsigned int m_fullscreenVBO = 0;

    unsigned int m_crosshairVAO = 0;
    unsigned int m_crosshairVBO = 0;

    int m_sceneMVP = -1;
    int m_sceneTexture = -1;
    int m_sceneTextured = -1;
    int m_sceneColorUniform = -1;
    int m_sceneUVRect = -1;
    int m_sceneFrost = -1;
    int m_sceneBlurTex = -1;
    int m_sceneScreen = -1;

    int m_blitTexture = -1;
    int m_blitAlpha = -1;

    int m_panoramaFwd = -1;
    int m_panoramaRight = -1;
    int m_panoramaUp = -1;
    int m_panoramaTanX = -1;
    int m_panoramaTanY = -1;
    int m_panoramaLod = -1;
    int m_panoramaSampler = -1;

    std::string m_panoramaPath;   // requested (resolved) path
    std::string m_panoramaLoaded; // path actually loaded into m_panoramaTex
    std::filesystem::file_time_type m_panoramaMtime{};
    bool m_panoramaMtimeValid = false;
    unsigned int m_panoramaTex = 0;
    int m_panoramaW = 0;

    // Backdrop blur (decoration:blur). Two half-res ping-pong targets; the
    // window shader samples the final one at screen coordinates for pixels
    // where the client content is transparent.
    unsigned int m_blurFBOA = 0, m_blurTexA = 0;
    unsigned int m_blurFBOB = 0, m_blurTexB = 0;
    unsigned int m_blurProgram = 0;
    int m_blurUTex = -1, m_blurUTexel = -1, m_blurUDir = -1;
    int m_blurURadius = -1, m_blurUVibrancy = -1, m_blurUFinal = -1;
    int  m_blurW = 0, m_blurH = 0;
    unsigned int m_blurFinalTex = 0;
    bool m_blurEnabled  = false;
    int  m_blurSize     = 3;
    int  m_blurPasses   = 1;
    float m_blurVibrancy = 0.1696f;
    bool m_useBlur      = false;
    bool m_frost        = false;
    int  m_width = 0, m_height = 0;

    // Single source of truth for the view pose: mouse-look, movement and the
    // crosshair pick all read these same yaw/pitch values.
    Camera m_camera;

    float m_time = 0.0f;

    bool          m_probeRequested = false;
    bool          m_probeValid     = false;
    unsigned char m_probe[4]       = {0, 0, 0, 0};
};

} // namespace H3D
