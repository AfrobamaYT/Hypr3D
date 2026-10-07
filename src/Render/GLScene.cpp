#include "GLScene.hpp"

#include "../../third_party/font8x8_basic.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <GLES3/gl32.h>

// The test stubs carry only a subset of the GL constants.
#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif

#include <hyprgraphics/image/Image.hpp>

#include <algorithm>
#include <array>
#include <utility>
#include <functional>
#include <memory>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <system_error>
#include <vector>
#include <iterator>

namespace H3D {

namespace {

constexpr float PI = 3.14159265358979323846f;

// Real ground plane. The camera is constrained above this level and the grid
// is drawn on this plane, not on a ceiling. Windows live above it.
constexpr float FLOOR_Y = Camera::kFloorY;

// --- window transparency: BSP-ordered exact compositing ---------------------
//
// Two translucent window quads crossing in space cannot be blended correctly
// by a single global draw order: per pixel, the draw order must match which
// plane is actually closer, and for crossing planes that flips across the
// intersection line. A BSP built from the window planes splits crossing quads
// along those lines and the back-to-front traversal yields a per-pixel
// correct order (the classic use of BSP for alpha sorting). Every window is
// a handful of polygons, so the tree is tiny.

struct SWVert {
    Vec3  p{};
    float u = 0.f, v = 0.f;
};

struct SWPoly {
    std::vector<SWVert> verts;
    unsigned int        tex = 0;
    float               alpha = 1.0f; // the window's room alpha (g_fsFade etc.)
};

// Split a polygon by a plane. Pieces on the positive side go to `front`,
// negative to `back`; vertices on the plane land in both.
static void bspSplitPoly(const SWPoly& poly, const Vec3& pl, const Vec3& n,
                         SWPoly& front, SWPoly& back,
                         bool& hasFront, bool& hasBack) {
    front.verts.clear();
    back.verts.clear();
    // Both pieces belong to the same window: carry its texture and alpha,
    // or a split piece renders with texture 0 -- pure black.
    front.tex      = poly.tex;
    front.alpha    = poly.alpha;
    back.tex       = poly.tex;
    back.alpha     = poly.alpha;
    hasFront = hasBack = false;

    const size_t COUNT = poly.verts.size();
    if (COUNT < 3 || COUNT > 32)
        return;

    float dist[32];
    for (size_t i = 0; i < COUNT; ++i)
        dist[i] = dot(poly.verts[i].p - pl, n);

    for (size_t i = 0; i < COUNT; ++i) {
        const size_t J = (i + 1) % COUNT;
        const auto&  A = poly.verts[i];
        const auto&  B = poly.verts[J];
        const float  DA = dist[i], DB = dist[J];

        if (DA >= 0.f) {
            front.verts.push_back(A);
            if (DA > 0.f)
                hasFront = true;
        }
        if (DA <= 0.f) {
            back.verts.push_back(A);
            if (DA < 0.f)
                hasBack = true;
        }

        if ((DA > 0.f && DB < 0.f) || (DA < 0.f && DB > 0.f)) {
            const float T = DA / (DA - DB);
            SWVert I{
                A.p + (B.p - A.p) * T,
                A.u + (B.u - A.u) * T,
                A.v + (B.v - A.v) * T,
            };
            front.verts.push_back(I);
            back.verts.push_back(I);
        }
    }

    hasFront = hasFront && front.verts.size() >= 3;
    hasBack  = hasBack  && back.verts.size()  >= 3;
}

struct SBspNode {
    Vec3 p{}, n{};
    std::vector<SWPoly>        coplanar;
    std::unique_ptr<SBspNode>  front, back;
};

static void bspBuild(SBspNode& node, std::vector<SWPoly>& polys, int depth) {
    if (polys.empty())
        return;

    // Node plane from the first polygon.
    const Vec3 A = polys[0].verts[0].p;
    const Vec3 B = polys[0].verts[1].p;
    const Vec3 C = polys[0].verts[2].p;
    node.p = A;
    node.n = normalize(cross(B - A, C - A));

    std::vector<SWPoly> frontList, backList;

    for (auto& POLY : polys) {
        float dist[32];
        const size_t COUNT = std::min<size_t>(POLY.verts.size(), 32);

        bool pos = false, neg = false;
        for (size_t i = 0; i < COUNT; ++i) {
            dist[i] = dot(POLY.verts[i].p - node.p, node.n);
            pos = pos || dist[i] > 1e-4f;
            neg = neg || dist[i] < -1e-4f;
        }

        if (!pos) {
            node.coplanar.push_back(std::move(POLY));
        } else if (!neg) {
            frontList.push_back(std::move(POLY));
        } else {
            SWPoly front, back;
            bool hasFront = false, hasBack = false;
            bspSplitPoly(POLY, node.p, node.n, front, back, hasFront, hasBack);

            if (hasFront)
                frontList.push_back(std::move(front));
            if (hasBack)
                backList.push_back(std::move(back));
            // Fully straddling polygons lose nothing: both pieces carry the
            // texture onward.
        }
    }

    constexpr int MAX_DEPTH = 12;

    if (depth < MAX_DEPTH && !frontList.empty()) {
        node.front = std::make_unique<SBspNode>();
        bspBuild(*node.front, frontList, depth + 1);
    } else {
        for (auto& P : frontList)
            node.coplanar.push_back(std::move(P));
    }

    if (depth < MAX_DEPTH && !backList.empty()) {
        node.back = std::make_unique<SBspNode>();
        bspBuild(*node.back, backList, depth + 1);
    } else {
        for (auto& P : backList)
            node.coplanar.push_back(std::move(P));
    }
}

static void bspTraverse(const SBspNode& node, const Vec3& eye,
                        const std::function<void(const SWPoly&)>& emit) {
    const float SIDE = dot(eye - node.p, node.n);

    // The side the eye is on is NEARER: draw the other side first.
    const SBspNode* FAR_FIRST = SIDE > 0 ? node.back.get() : node.front.get();
    const SBspNode* NEAR_LAST = SIDE > 0 ? node.front.get() : node.back.get();

    if (FAR_FIRST)
        bspTraverse(*FAR_FIRST, eye, emit);

    for (const auto& P : node.coplanar)
        emit(P);

    if (NEAR_LAST)
        bspTraverse(*NEAR_LAST, eye, emit);
}

static GLuint compileShader(
    GLenum type,
    const char* source
) {
    const GLuint shader = glCreateShader(type);

    if (!shader)
        return 0;

    glShaderSource(
        shader,
        1,
        &source,
        nullptr
    );

    glCompileShader(shader);

    GLint compiled = GL_FALSE;

    glGetShaderiv(
        shader,
        GL_COMPILE_STATUS,
        &compiled
    );

    if (compiled == GL_TRUE)
        return shader;

    glDeleteShader(shader);

    return 0;
}

static GLuint linkProgram(
    GLuint vertexShader,
    GLuint fragmentShader
) {
    const GLuint program = glCreateProgram();

    if (!program)
        return 0;

    glAttachShader(
        program,
        vertexShader
    );

    glAttachShader(
        program,
        fragmentShader
    );

    glLinkProgram(program);

    GLint linked = GL_FALSE;

    glGetProgramiv(
        program,
        GL_LINK_STATUS,
        &linked
    );

    if (linked == GL_TRUE)
        return program;

    glDeleteProgram(program);

    return 0;
}

static void setupMeshVAO(
    GLuint vao,
    GLuint vbo,
    const std::vector<float>& vertices
) {
    glBindVertexArray(vao);

    glBindBuffer(
        GL_ARRAY_BUFFER,
        vbo
    );

    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(
            vertices.size() * sizeof(float)
        ),
        vertices.data(),
        GL_STATIC_DRAW
    );

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        5 * sizeof(float),
        reinterpret_cast<void*>(0)
    );

    glEnableVertexAttribArray(1);

    glVertexAttribPointer(
        1,
        2,
        GL_FLOAT,
        GL_FALSE,
        5 * sizeof(float),
        reinterpret_cast<void*>(
            3 * sizeof(float)
        )
    );

    glBindVertexArray(0);
}

static void appendVertex(
    std::vector<float>& out,
    float x,
    float y,
    float z,
    float u,
    float v
) {
    out.push_back(x);
    out.push_back(y);
    out.push_back(z);
    out.push_back(u);
    out.push_back(v);
}

static void appendTriangle(
    std::vector<float>& out,
    float x0, float y0, float z0, float u0, float v0,
    float x1, float y1, float z1, float u1, float v1,
    float x2, float y2, float z2, float u2, float v2
) {
    appendVertex(out, x0, y0, z0, u0, v0);
    appendVertex(out, x1, y1, z1, u1, v1);
    appendVertex(out, x2, y2, z2, u2, v2);
}

// UV convention this scene relies on: u grows with +X, v grows with +Y.
// A window quad therefore samples (u0,v0) at its bottom-left and (u1,v1) at
// its top-right, which is why the caller flips logical Y when computing the
// subrect (Hyprland logical coordinates have their origin top-left).
static void appendQuad(
    std::vector<float>& out,
    float x0, float y0, float z0,
    float x1, float y1, float z1,
    float x2, float y2, float z2,
    float x3, float y3, float z3
) {
    appendTriangle(
        out,
        x0, y0, z0, 0.0f, 0.0f,
        x1, y1, z1, 1.0f, 0.0f,
        x2, y2, z2, 1.0f, 1.0f
    );

    appendTriangle(
        out,
        x0, y0, z0, 0.0f, 0.0f,
        x2, y2, z2, 1.0f, 1.0f,
        x3, y3, z3, 0.0f, 1.0f
    );
}

} // namespace

GLScene::GLScene() {
    reset();
}

bool GLScene::initialize() {
    if (m_initialized)
        return true;

    if (!createPrograms())
        return false;

    if (!createMeshes())
        return false;

    m_initialized = true;

    return true;
}

bool GLScene::createPrograms() {
    static constexpr const char* sceneVertexShader = R"GLSL(
#version 300 es

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aUV;

uniform mat4 uMVP;

// xy = subrect origin, zw = subrect size. Keeps the whole scene on one
// program: untextured geometry just passes (0,0,1,1).
uniform vec4 uUVRect;

out vec2 vUV;

void main() {
    gl_Position = uMVP * vec4(aPosition, 1.0);
    vUV = aUV * uUVRect.zw + uUVRect.xy;
}
)GLSL";

    static constexpr const char* sceneFragmentShader = R"GLSL(
#version 300 es

precision mediump float;

in vec2 vUV;

uniform sampler2D uTexture;
uniform int uTextured;
uniform vec4 uColor;
// Windows only (drawWindows), 0 for everything else.
uniform float uLodBias;
// Portals only: pixels less opaque than this are cut out, 0 for the rest.
uniform float uAlphaCut;

out vec4 fragColor;

void main() {
    if (uTextured != 0)
        fragColor = texture(uTexture, vUV, uLodBias) * uColor;
    else
        fragColor = uColor;
    if (fragColor.a < uAlphaCut)
        discard;
}
)GLSL";

    static constexpr const char* blitVertexShader = R"GLSL(
#version 300 es

layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aUV;

out vec2 vUV;

void main() {
    gl_Position = vec4(
        aPosition,
        0.0,
        1.0
    );

    vUV = aUV;
}
)GLSL";

    static constexpr const char* blitFragmentShader = R"GLSL(
#version 300 es

// highp: the flip below is arithmetic, and the RTX 3080 runs mediump
// arithmetic as 16-bit floats -- 1.0 - v came out in steps of about half
// a row of a 1080-row scene, a sawtooth that smeared every horizontal
// edge (5.3 % of a terminal's pixels off, measured 2026-10-07).
precision highp float;

in vec2 vUV;

uniform sampler2D uTexture;
uniform float uAlpha;

out vec4 fragColor;

void main() {
    // Hyprland framebuffers are vertically flipped relative to the GL
    // convention: their projection maps logical Y (growing DOWN) into NDC
    // without negation, so NDC y = -1 is the TOP of the screen (the DRM
    // scanout compensates). Our scene FBO is rendered with the standard
    // convention (NDC -1 = scene bottom = texture v 0). Sampling with a
    // flipped v puts the scene upright on screen; without it the whole
    // world renders mirrored vertically -- the floor reads as a ceiling,
    // window hover lands on the opposite half of the window, and the
    // camera look feels inverted.
    vec2 uv = vec2(vUV.x, 1.0 - vUV.y);

    vec4 color = texture(
        uTexture,
        uv
    );

    // The scene itself is opaque. uAlpha is only the 2D->3D transition; it
    // must not be multiplied by per-pixel scene alpha left behind by grid or
    // window texture blending.
    fragColor = vec4(
        color.rgb,
        uAlpha
    );
}
)GLSL";

    const GLuint sceneVS =
        compileShader(
            GL_VERTEX_SHADER,
            sceneVertexShader
        );

    if (!sceneVS)
        return false;

    const GLuint sceneFS =
        compileShader(
            GL_FRAGMENT_SHADER,
            sceneFragmentShader
        );

    if (!sceneFS) {
        glDeleteShader(sceneVS);
        return false;
    }

    m_sceneProgram =
        linkProgram(
            sceneVS,
            sceneFS
        );

    glDeleteShader(sceneVS);
    glDeleteShader(sceneFS);

    if (!m_sceneProgram)
        return false;

    const GLuint blitVS =
        compileShader(
            GL_VERTEX_SHADER,
            blitVertexShader
        );

    if (!blitVS)
        return false;

    const GLuint blitFS =
        compileShader(
            GL_FRAGMENT_SHADER,
            blitFragmentShader
        );

    if (!blitFS) {
        glDeleteShader(blitVS);
        return false;
    }

    m_blitProgram =
        linkProgram(
            blitVS,
            blitFS
        );

    glDeleteShader(blitVS);
    glDeleteShader(blitFS);

    if (!m_blitProgram)
        return false;

    static constexpr const char* panoramaVertexShader = R"GLSL(
#version 300 es

layout(location = 0) in vec2 aPosition;

out vec2 vNdc;

void main() {
    vNdc = aPosition;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
)GLSL";

    static constexpr const char* panoramaFragmentShader = R"GLSL(
#version 300 es

// highp is mandatory here: mediump cannot represent the longitude with texel
// precision, and the atan2 discontinuity at the +-pi wrap point then shows up
// as a one-pixel seam.
precision highp float;

in vec2 vNdc;

uniform vec3 uFwd;
uniform vec3 uRight;
uniform vec3 uUp;
uniform float uTanHalfX;
uniform float uTanHalfY;
uniform vec2 uTanCenter; // the rectangle's centre: off-axis on a side monitor
uniform float uLod;
uniform sampler2D uPanorama;

out vec4 fragColor;

void main() {
    // Reconstruct the view ray for this pixel from the camera basis, then
    // sample the equirectangular panorama. fract() keeps u strictly inside
    // [0, 1) so the wrap point never rides the texture's outer edge.
    vec3 dir = normalize(
        uFwd + uRight * (uTanCenter.x + vNdc.x * uTanHalfX) +
        uUp * (uTanCenter.y + vNdc.y * uTanHalfY));

    float lon = atan(dir.x, -dir.z);
    float lat = asin(clamp(dir.y, -1.0, 1.0));

    vec2 uv = vec2(
        fract(0.5 + lon / 6.28318530718),
        0.5 - lat / 3.14159265359);

    // textureLod with an analytically computed level: the implicit derivative
    // of u explodes at the atan2 wrap point, which made the GPU pick a coarse
    // mip for exactly one pixel -- the seam. An explicit LOD never spikes.
    fragColor = vec4(textureLod(uPanorama, uv, uLod).rgb, 1.0);
}
)GLSL";

    const GLuint PANORAMA_VS =
        compileShader(
            GL_VERTEX_SHADER,
            panoramaVertexShader
        );

    if (!PANORAMA_VS)
        return false;

    const GLuint PANORAMA_FS =
        compileShader(
            GL_FRAGMENT_SHADER,
            panoramaFragmentShader
        );

    if (!PANORAMA_FS) {
        glDeleteShader(PANORAMA_VS);
        return false;
    }

    m_panoramaProgram =
        linkProgram(
            PANORAMA_VS,
            PANORAMA_FS
        );

    glDeleteShader(PANORAMA_VS);
    glDeleteShader(PANORAMA_FS);

    if (!m_panoramaProgram)
        return false;

    m_sceneMVP =
        glGetUniformLocation(
            m_sceneProgram,
            "uMVP"
        );

    m_sceneTexture =
        glGetUniformLocation(
            m_sceneProgram,
            "uTexture"
        );

    m_sceneTextured =
        glGetUniformLocation(
            m_sceneProgram,
            "uTextured"
        );

    m_sceneColorUniform =
        glGetUniformLocation(
            m_sceneProgram,
            "uColor"
        );

    m_sceneUVRect =
        glGetUniformLocation(
            m_sceneProgram,
            "uUVRect"
        );

    m_sceneLodBias = glGetUniformLocation(m_sceneProgram, "uLodBias");
    m_sceneAlphaCut = glGetUniformLocation(m_sceneProgram, "uAlphaCut");

    m_blitTexture =
        glGetUniformLocation(
            m_blitProgram,
            "uTexture"
        );

    m_blitAlpha =
        glGetUniformLocation(
            m_blitProgram,
            "uAlpha"
        );

    m_panoramaFwd =
        glGetUniformLocation(
            m_panoramaProgram,
            "uFwd"
        );

    m_panoramaRight =
        glGetUniformLocation(
            m_panoramaProgram,
            "uRight"
        );

    m_panoramaUp =
        glGetUniformLocation(
            m_panoramaProgram,
            "uUp"
        );

    m_panoramaTanX =
        glGetUniformLocation(
            m_panoramaProgram,
            "uTanHalfX"
        );

    m_panoramaTanY =
        glGetUniformLocation(
            m_panoramaProgram,
            "uTanHalfY"
        );

    m_panoramaTanCenter =
        glGetUniformLocation(
            m_panoramaProgram,
            "uTanCenter"
        );

    m_panoramaLod =
        glGetUniformLocation(
            m_panoramaProgram,
            "uLod"
        );

    m_panoramaSampler =
        glGetUniformLocation(
            m_panoramaProgram,
            "uPanorama"
        );

    return
        m_sceneMVP >= 0 &&
        m_sceneTexture >= 0 &&
        m_sceneTextured >= 0 &&
        m_sceneColorUniform >= 0 &&
        m_sceneUVRect >= 0 &&
        m_blitTexture >= 0 &&
        m_blitAlpha >= 0 &&
        m_panoramaFwd >= 0 &&
        m_panoramaRight >= 0 &&
        m_panoramaUp >= 0 &&
        m_panoramaTanX >= 0 &&
        m_panoramaTanY >= 0 &&
        m_panoramaTanCenter >= 0 &&
        m_panoramaLod >= 0 &&
        m_panoramaSampler >= 0;
}

bool GLScene::createMeshes() {
    // Unit window quad, centred on the origin, facing +Z (the pose windows
    // are modelled in). Per-window position and size come from the model
    // matrix, so every window is independently placeable and scalable.
    std::vector<float> quad;

    appendQuad(
        quad,
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.5f,  0.5f, 0.0f,
        -0.5f,  0.5f, 0.0f
    );

    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &m_quadVBO);

    setupMeshVAO(m_quadVAO, m_quadVBO, quad);

    m_quadVertexCount =
        static_cast<int>(quad.size() / 5);

    // Ground quad lying in the XZ plane.
    std::vector<float> floor;

    // Counter-clockwise when viewed from above: +Y is the floor normal.
    appendQuad(
        floor,
        -0.5f, 0.0f,  0.5f,
         0.5f, 0.0f,  0.5f,
         0.5f, 0.0f, -0.5f,
        -0.5f, 0.0f, -0.5f
    );

    glGenVertexArrays(1, &m_floorVAO);
    glGenBuffers(1, &m_floorVBO);

    setupMeshVAO(m_floorVAO, m_floorVBO, floor);

    m_floorVertexCount =
        static_cast<int>(floor.size() / 5);

    std::vector<float> grid;

    constexpr int   gridMin = -20;
    constexpr int   gridMax = 20;

    for (int i = gridMin; i <= gridMax; ++i) {
        const float p = static_cast<float>(i);

        grid.push_back(static_cast<float>(gridMin));
        grid.push_back(FLOOR_Y);
        grid.push_back(p);

        grid.push_back(static_cast<float>(gridMax));
        grid.push_back(FLOOR_Y);
        grid.push_back(p);

        grid.push_back(p);
        grid.push_back(FLOOR_Y);
        grid.push_back(static_cast<float>(gridMin));

        grid.push_back(p);
        grid.push_back(FLOOR_Y);
        grid.push_back(static_cast<float>(gridMax));
    }

    glGenVertexArrays(1, &m_gridVAO);
    glGenBuffers(1, &m_gridVBO);

    glBindVertexArray(m_gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_gridVBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(grid.size() * sizeof(float)),
        grid.data(),
        GL_STATIC_DRAW
    );

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        3 * sizeof(float),
        reinterpret_cast<void*>(0)
    );

    glDisableVertexAttribArray(1);

    glBindVertexArray(0);

    m_gridVertexCount =
        static_cast<int>(grid.size() / 3);

    const float fullscreen[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,

        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };

    glGenVertexArrays(1, &m_fullscreenVAO);
    glGenBuffers(1, &m_fullscreenVBO);

    glBindVertexArray(m_fullscreenVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_fullscreenVBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(fullscreen),
        fullscreen,
        GL_STATIC_DRAW
    );

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        reinterpret_cast<void*>(0)
    );

    glEnableVertexAttribArray(1);

    glVertexAttribPointer(
        1,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        reinterpret_cast<void*>(2 * sizeof(float))
    );

    glBindVertexArray(0);

    // Dynamic screen-space crosshair. Four independent rectangular arms are
    // used instead of GL_LINES so the shape stays visibly cross-like and its
    // thickness remains stable across drivers.
    glGenVertexArrays(1, &m_crosshairVAO);
    glGenBuffers(1, &m_crosshairVBO);

    glBindVertexArray(m_crosshairVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_crosshairVBO);

    const std::vector<float> crosshair(24 * 5, 0.0f);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(crosshair.size() * sizeof(float)),
        crosshair.data(),
        GL_DYNAMIC_DRAW
    );

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE,
        5 * sizeof(float), reinterpret_cast<void*>(0)
    );

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 2, GL_FLOAT, GL_FALSE,
        5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float))
    );

    glBindVertexArray(0);

    return true;
}

bool GLScene::ensureSceneFramebuffer(
    int width,
    int height
) {
    if (
        m_sceneFBO &&
        m_sceneWidth == width &&
        m_sceneHeight == height
    )
        return true;

    // Another monitor's size: swap in the target kept for it, or keep the
    // current one aside and make a new one, instead of reallocating.
    if (m_sceneFBO) {
        const SceneTarget CURRENT{m_sceneWidth, m_sceneHeight, m_sceneFBO, m_sceneColor, m_sceneDepth};
        const auto KEPT = std::find_if(m_otherSceneTargets.begin(), m_otherSceneTargets.end(),
            [&](const SceneTarget& t) { return t.width == width && t.height == height; });

        if (KEPT != m_otherSceneTargets.end()) {
            m_sceneWidth  = KEPT->width;
            m_sceneHeight = KEPT->height;
            m_sceneFBO    = KEPT->fbo;
            m_sceneColor  = KEPT->color;
            m_sceneDepth  = KEPT->depth;
            *KEPT         = CURRENT;
            return true;
        }

        // Bounded: monitors come and go; the oldest size is released.
        if (m_otherSceneTargets.size() >= 4) {
            auto& OLD = m_otherSceneTargets.front();
            glDeleteRenderbuffers(1, &OLD.depth);
            glDeleteTextures(1, &OLD.color);
            glDeleteFramebuffers(1, &OLD.fbo);
            m_otherSceneTargets.erase(m_otherSceneTargets.begin());
        }

        m_otherSceneTargets.push_back(CURRENT);
        m_sceneFBO   = 0;
        m_sceneColor = 0;
        m_sceneDepth = 0;
    }

    if (!m_sceneFBO) {
        glGenFramebuffers(1, &m_sceneFBO);
        glGenTextures(1, &m_sceneColor);
        glGenRenderbuffers(1, &m_sceneDepth);
    }

    m_sceneWidth = width;
    m_sceneHeight = height;

    glBindTexture(GL_TEXTURE_2D, m_sceneColor);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA8,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        nullptr
    );

    glBindTexture(GL_TEXTURE_2D, 0);

    glBindRenderbuffer(GL_RENDERBUFFER, m_sceneDepth);

    glRenderbufferStorage(
        GL_RENDERBUFFER,
        GL_DEPTH_COMPONENT24,
        width,
        height
    );

    glBindFramebuffer(GL_FRAMEBUFFER, m_sceneFBO);

    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        m_sceneColor,
        0
    );

    glFramebufferRenderbuffer(
        GL_FRAMEBUFFER,
        GL_DEPTH_ATTACHMENT,
        GL_RENDERBUFFER,
        m_sceneDepth
    );

    const GLenum drawBuffer = GL_COLOR_ATTACHMENT0;

    glDrawBuffers(1, &drawBuffer);

    const GLenum status =
        glCheckFramebufferStatus(GL_FRAMEBUFFER);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return status == GL_FRAMEBUFFER_COMPLETE;
}

void GLScene::drawQuad(
    unsigned int vao,
    int vertexCount,
    const Mat4& mvp,
    unsigned int texture,
    const float uvRect[4],
    float r, float g, float b, float a
) {
    glUseProgram(m_sceneProgram);

    glUniformMatrix4fv(
        m_sceneMVP,
        1,
        GL_FALSE,
        mvp.m.data()
    );

    glUniform1i(m_sceneTextured, texture ? 1 : 0);

    glUniform4f(m_sceneColorUniform, r, g, b, a);

    glUniform4f(
        m_sceneUVRect,
        uvRect[0],
        uvRect[1],
        uvRect[2] - uvRect[0],
        uvRect[3] - uvRect[1]
    );

    if (texture) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(m_sceneTexture, 0);
    }

    glBindVertexArray(vao);

    glDrawArrays(GL_TRIANGLES, 0, vertexCount);

    glBindVertexArray(0);

    if (texture)
        glBindTexture(GL_TEXTURE_2D, 0);
}

void GLScene::drawFloor(
    const Mat4& vp
) {
    // Solid ground just under the grid. The floor is deliberately single-sided:
    // seeing this plane from below is a ceiling by definition, so back-face
    // culling makes the world coordinate system visually unambiguous.
    // The platform is centered on world zero and matches the grid lines'
    // extent (both halves of the visual and the collision slab in main.cpp).
    const Mat4 model =
        Mat4::translation({0.0f, FLOOR_Y - 0.02f, 0.0f}) *
        Mat4::scale({40.0f, 1.0f, 40.0f});

    static constexpr float fullUV[4] = {0.0f, 0.0f, 1.0f, 1.0f};

    GLint oldCullMode = GL_BACK;
    GLint oldFrontFace = GL_CCW;
    GLboolean oldCull = GL_FALSE;

    glGetBooleanv(GL_CULL_FACE, &oldCull);
    glGetIntegerv(GL_CULL_FACE_MODE, &oldCullMode);
    glGetIntegerv(GL_FRONT_FACE, &oldFrontFace);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    drawQuad(
        m_floorVAO,
        m_floorVertexCount,
        vp * model,
        0,
        fullUV,
        0.018f, 0.026f, 0.040f, 1.0f
    );

    glCullFace(static_cast<GLenum>(oldCullMode));
    glFrontFace(static_cast<GLenum>(oldFrontFace));

    if (oldCull)
        glEnable(GL_CULL_FACE);
    else
        glDisable(GL_CULL_FACE);
}

void GLScene::refreshPanorama() {
    if (m_panoramaPath.empty()) {
        if (m_panoramaTex) {
            glDeleteTextures(1, &m_panoramaTex);
            m_panoramaTex = 0;
            m_panoramaLoaded.clear();
            m_panoramaMtimeValid = false;
        }
        return;
    }

    std::error_code ec;
    const auto MTIME = std::filesystem::last_write_time(m_panoramaPath, ec);

    if (!ec && m_panoramaMtimeValid && m_panoramaTex &&
        MTIME == m_panoramaMtime && m_panoramaLoaded == m_panoramaPath)
        return; // unchanged since the last load

    m_panoramaMtime      = MTIME;
    m_panoramaMtimeValid = !ec;
    m_panoramaLoaded     = m_panoramaPath;

    Hyprgraphics::CImage image(m_panoramaPath);

    auto surface = image.success() ? image.cairoSurface() : nullptr;

    if (!surface || surface->status() != CAIRO_STATUS_SUCCESS) {
        // Drop the old texture so the fallback void shows through.
        if (m_panoramaTex) {
            glDeleteTextures(1, &m_panoramaTex);
            m_panoramaTex = 0;
        }
        return;
    }

    const int W      = static_cast<int>(surface->size().x);
    const int H      = static_cast<int>(surface->size().y);
    const int STRIDE = surface->stride();
    const auto* SRC  = surface->data();

    if (W <= 0 || H <= 0 || !SRC) {
        if (m_panoramaTex) {
            glDeleteTextures(1, &m_panoramaTex);
            m_panoramaTex = 0;
        }
        return;
    }

    // Cairo ARGB32 is premultiplied BGRA in memory; the panorama background
    // is opaque, so a straight channel swap to RGBA is the whole conversion.
    std::vector<unsigned char> pixels(static_cast<size_t>(W) * H * 4);

    for (int y = 0; y < H; ++y) {
        const auto* row = SRC + static_cast<size_t>(y) * STRIDE;
        auto* dst = pixels.data() + static_cast<size_t>(y) * W * 4;

        for (int x = 0; x < W; ++x) {
            dst[x * 4 + 0] = row[x * 4 + 2];
            dst[x * 4 + 1] = row[x * 4 + 1];
            dst[x * 4 + 2] = row[x * 4 + 0];
            dst[x * 4 + 3] = 255;
        }
    }

    if (m_panoramaTex)
        glDeleteTextures(1, &m_panoramaTex);

    glGenTextures(1, &m_panoramaTex);

    glBindTexture(GL_TEXTURE_2D, m_panoramaTex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);

    m_panoramaW = W;
}

// Diff-apply the config object list against the slot vector.
void GLScene::setSceneObjects(const std::vector<SSceneSpec>& specs) {
    while (m_slots.size() > specs.size()) {
        if (m_slots.back().model)
            m_slots.back().model->destroy();
        m_slots.pop_back();
    }

    m_slots.resize(specs.size());

    for (size_t i = 0; i < specs.size(); ++i) {
        auto& S = m_slots[i];

        if (!S.model)
            S.model = std::make_unique<CMapModel>();

        const bool TRANSFORM_CHANGED =
            S.spec.position.x != specs[i].position.x ||
            S.spec.position.y != specs[i].position.y ||
            S.spec.position.z != specs[i].position.z ||
            S.spec.rotationDeg.x != specs[i].rotationDeg.x ||
            S.spec.rotationDeg.y != specs[i].rotationDeg.y ||
            S.spec.rotationDeg.z != specs[i].rotationDeg.z ||
            S.spec.scale.x != specs[i].scale.x ||
            S.spec.scale.y != specs[i].scale.y ||
            S.spec.scale.z != specs[i].scale.z;

        const bool CENTER_CHANGED =
            S.spec.center != specs[i].center ||
            S.spec.centerOffset.x != specs[i].centerOffset.x ||
            S.spec.centerOffset.y != specs[i].centerOffset.y ||
            S.spec.centerOffset.z != specs[i].centerOffset.z;

        S.spec = specs[i];

        S.model->setEmissiveScale(S.spec.emissiveScale);
        S.model->setFlat(S.spec.flat);

        if (CENTER_CHANGED)
            S.model->setCenter(S.spec.center, S.spec.centerOffset);

        if (TRANSFORM_CHANGED)
            S.model->setTransform(S.spec.position, S.spec.rotationDeg,
                                  S.spec.scale);
    }
}

void GLScene::setSceneObjectTransform(size_t index, const Vec3& position,
                                      const Vec3& rotationDeg) {
    if (index >= m_slots.size() || !m_slots[index].model)
        return;

    auto& S = m_slots[index];
    S.spec.position    = position;
    S.spec.rotationDeg = rotationDeg;
    S.model->setTransform(S.spec.position, S.spec.rotationDeg, S.spec.scale);
}

uint64_t GLScene::sceneFingerprint() const {
    // Fold per-model generations + slot count: any (re)load, transform
    // change, or add/remove changes the fingerprint.
    uint64_t F = 1469598103934665603ull;

    const auto MIX = [&](uint64_t V) {
        F ^= V;
        F *= 1099511628211ull;
    };

    MIX(m_slots.size());

    for (const auto& S : m_slots)
        MIX(S.model ? S.model->generation() : 0ull);

    return F;
}

// Loads (or reloads) each scene object when its config path or the file's
// mtime changed. Runs inside render() so the EGL context is current -- the
// same rule as the panorama.
void GLScene::refreshScene() {
    for (auto& S : m_slots) {
        // A file read and decoded on the model's worker gets its GL objects
        // here, with the context current.
        if (S.model)
            S.model->poll();

        const std::string& CFG_PATH = S.spec.path;

        if (CFG_PATH.empty()) {
            if (S.model && (S.model->loaded() || S.model->pending())) {
                S.model->destroy();
                S.loadedPath.clear();
                S.mtimeValid = false;
            }
            continue;
        }

        // Expand a leading tilde like the panorama path does -- config
        // paths arrive as "~/..." and cgltf would look for a literal '~'.
        std::string path = CFG_PATH;
        if (path.starts_with('~')) {
            if (const char* HOME = getenv("HOME"))
                path = std::string{HOME} + path.substr(1);
        }

        std::error_code ec;
        const auto MTIME = std::filesystem::last_write_time(path, ec);

        // The same file as last time, also when it is still missing. Loading,
        // loaded or failed, it is not started again until it changes: a load
        // per frame would start a worker per frame.
        const bool SAME_FILE = S.loadedPath == path &&
            S.mtimeValid == !ec && (ec || MTIME == S.mtime);
        const bool UNCHANGED = S.model && SAME_FILE &&
            (S.model->loaded() || S.model->pending() || S.model->failed());

        if (UNCHANGED)
            continue;

        S.mtime      = MTIME;
        S.mtimeValid = !ec;
        S.loadedPath = path;

        S.model->load(path, S.spec.position, S.spec.rotationDeg,
                      S.spec.scale);
    }
}

// A capsule outline at a body center: three circles + four verticals,
// rebuilt every drawn frame (a couple hundred floats). With a yaw, a line
// from the head shows where it looks. The player's collision capsule (F3)
// draws over everything; the companion is hidden behind what stands in
// front of it.
void GLScene::drawCapsule(const Mat4& vp, const Vec3& center, const float* yaw,
                          const Vec3& color, bool depthTest) {
    if (!m_pDbgProgram) {
        static const char* VS = R"GLSL(
#version 320 es
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
)GLSL";
        static const char* FS = R"GLSL(
#version 320 es
precision mediump float;
uniform vec3 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, 1.0); }
)GLSL";
        const GLuint VSx = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(VSx, 1, &VS, nullptr);
        glCompileShader(VSx);
        const GLuint FSx = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(FSx, 1, &FS, nullptr);
        glCompileShader(FSx);
        const GLuint P = glCreateProgram();
        glAttachShader(P, VSx);
        glAttachShader(P, FSx);
        glLinkProgram(P);
        glDeleteShader(VSx);
        glDeleteShader(FSx);
        m_pDbgProgram = P;
        m_pDbgMVP     = glGetUniformLocation(P, "uMVP");
        m_pDbgColor   = glGetUniformLocation(P, "uColor");
    }

    const float R = 0.3f;             // Camera::kBodyHalfWidth
    const float CY = center.y;        // the Jolt body center (feet + 0.9)
    const float HH = 0.6f;            // the cylinder half height (0.9 - r)
    const int SEG = 24;

    std::vector<float> V;
    V.reserve(3 * (3 * SEG * 2 + 8));
    const auto CIRCLE = [&](float y) {
        for (int s = 0; s < SEG; ++s) {
            const float A0 = float(s) / SEG * 6.2831853f;
            const float A1 = float(s + 1) / SEG * 6.2831853f;
            V.push_back(center.x + std::cos(A0) * R);
            V.push_back(CY + y);
            V.push_back(center.z + std::sin(A0) * R);
            V.push_back(center.x + std::cos(A1) * R);
            V.push_back(CY + y);
            V.push_back(center.z + std::sin(A1) * R);
        }
    };
    CIRCLE(-HH);
    CIRCLE(0.f);
    CIRCLE(HH);
    for (int s = 0; s < 4; ++s) {
        const float A = float(s) / 4 * 6.2831853f + 0.3926991f;
        const float X = center.x + std::cos(A) * R;
        const float Z = center.z + std::sin(A) * R;
        V.push_back(X); V.push_back(CY - HH); V.push_back(Z);
        V.push_back(X); V.push_back(CY + HH); V.push_back(Z);
    }
    if (yaw) { // forward is {sin yaw, 0, -cos yaw}, as the camera's
        V.push_back(center.x); V.push_back(CY + HH); V.push_back(center.z);
        V.push_back(center.x + std::sin(*yaw) * 0.6f);
        V.push_back(CY + HH);
        V.push_back(center.z - std::cos(*yaw) * 0.6f);
    }
    m_pDbgVerts = static_cast<int>(V.size() / 3);

    if (!m_pDbgVAO) {
        glGenVertexArrays(1, &m_pDbgVAO);
        glGenBuffers(1, &m_pDbgVBO);
        glBindVertexArray(m_pDbgVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_pDbgVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                              reinterpret_cast<void*>(0));
        glBindVertexArray(0);
    }

    glUseProgram(m_pDbgProgram);
    glUniformMatrix4fv(m_pDbgMVP, 1, GL_FALSE, vp.m.data());
    glUniform3f(m_pDbgColor, color.x, color.y, color.z);
    glBindVertexArray(m_pDbgVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_pDbgVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(V.size() * sizeof(float)),
                 V.data(), GL_STREAM_DRAW);
    // Depth-tested lines also write depth: the floor is drawn after them
    // and would otherwise paint over everything below the horizon.
    if (!depthTest)
        glDisable(GL_DEPTH_TEST);
    glDepthMask(depthTest ? GL_TRUE : GL_FALSE);
    glDrawArrays(GL_LINES, 0, m_pDbgVerts);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// Loads (or reloads) the player character when the config path or the
// file's mtime changed; applies the per-state animation assignment after a
// successful load. Runs inside render() so the EGL context is current.
void GLScene::refreshPlayer() {
    const std::string& CFG_PATH = m_playerCfg.path;

    if (CFG_PATH.empty()) {
        if (m_player.loaded())
            m_player.destroy();
        m_playerPath.clear();
        return;
    }

    std::string path = CFG_PATH;
    if (path.starts_with('~')) {
        if (const char* HOME = getenv("HOME"))
            path = std::string{HOME} + path.substr(1);
    }

    std::error_code ec;
    const auto MTIME = std::filesystem::last_write_time(path, ec);

    const bool UNCHANGED = m_player.loaded() && m_playerPath == path &&
        !ec && m_playerMtimeValid && MTIME == m_playerMtime;
    if (UNCHANGED)
        return;

    m_playerPath       = path;
    m_playerMtime      = MTIME;
    m_playerMtimeValid = !ec;

    if (ec) {
        if (m_player.loaded())
            m_player.destroy();
        return;
    }

    if (m_player.load(path)) {
        for (int s = 0; s < CPlayerModel::kStateCount; ++s) {
            const auto ST = static_cast<CPlayerModel::EState>(s);
            if (!m_playerCfg.animName[s].empty())
                m_player.setAnim(ST, m_playerCfg.animName[s]);
            else
                m_player.setAnim(ST, m_playerCfg.animIdx[s]);
            m_player.setAnimSpeed(ST, m_playerCfg.animSpeed[s]);
        }
    }
}

void GLScene::drawPanorama(const ViewWindow& view) {
    if (!m_panoramaTex || !m_panoramaProgram)
        return;

    glUseProgram(m_panoramaProgram);

    Vec3 FWD   = m_camera.forward();
    Vec3 RIGHT = m_camera.right();
    Vec3 UP    = cross(RIGHT, FWD);
    // The rectangle this monitor sees (render() has applied the zoom): half
    // extents and centre in tangents, so a side monitor samples the part of
    // the sky beside the main one.
    const float TANX = (view.right - view.left) * 0.5f;
    const float TANY = (view.top - view.bottom) * 0.5f;
    const float CENX = (view.right + view.left) * 0.5f;
    const float CENY = (view.top + view.bottom) * 0.5f;

    // Roll (walk bob): rotate the pixel->ray basis around the view axis by
    // the SAME angle the view matrix tilts its up vector, or the panorama
    // stays level while the scene rolls. Matches Camera::view(): screen up
    // = UP*cos(r) + RIGHT*sin(r), screen right = RIGHT*cos(r) - UP*sin(r).
    const float RROLL = m_camera.roll;
    Vec3 RRIGHT = RIGHT * std::cos(RROLL) - UP * std::sin(RROLL);
    Vec3 RUP    = UP * std::cos(RROLL) + RIGHT * std::sin(RROLL);

    // The front third-person view looks BACK along the look axis: the view
    // matrix and the world flip with it, and the panorama's pixel->ray
    // basis must flip too (direction and screen right; the up stays).
    if (m_camera.mirrorView) {
        FWD   = FWD * -1.0f;
        RRIGHT = RRIGHT * -1.0f;
    }

    // Analytic mip level: texels per screen pixel at the view centre. The
    // panorama is W texels around 2*pi radians; one screen pixel spans about
    // 2*tanX / screenWidth radians.
    float lod = 0.0f;

    if (m_panoramaW > 0 && m_sceneWidth > 0) {
        const float texelsPerPixel =
            static_cast<float>(m_panoramaW) / (2.0f * PI) *
            (2.0f * TANX) / static_cast<float>(m_sceneWidth);

        lod = std::clamp(std::log2(std::max(texelsPerPixel, 0.03125f)), 0.0f, 12.0f);
    }

    glUniform3f(m_panoramaFwd, FWD.x, FWD.y, FWD.z);
    glUniform3f(m_panoramaRight, RRIGHT.x, RRIGHT.y, RRIGHT.z);
    glUniform3f(m_panoramaUp, RUP.x, RUP.y, RUP.z);
    glUniform1f(m_panoramaTanX, TANX);
    glUniform1f(m_panoramaTanY, TANY);
    glUniform2f(m_panoramaTanCenter, CENX, CENY);
    glUniform1f(m_panoramaLod, lod);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_panoramaTex);
    glUniform1i(m_panoramaSampler, 0);

    // Pure background: no depth interaction, everything else draws on top.
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glBindVertexArray(m_fullscreenVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLScene::drawGrid(
    const Mat4& vp
) {
    // Lines, not triangles, and translucent: the grid is the visual floor and
    // sits just above the solid ground quad so the two never z-fight.
    glUseProgram(m_sceneProgram);

    glUniformMatrix4fv(
        m_sceneMVP,
        1,
        GL_FALSE,
        vp.m.data()
    );

    glUniform1i(m_sceneTextured, 0);
    glUniform4f(m_sceneColorUniform, 0.08f, 0.18f, 0.30f, 0.62f);
    glUniform4f(m_sceneUVRect, 0.0f, 0.0f, 1.0f, 1.0f);

    glBindVertexArray(m_gridVAO);

    glDrawArrays(GL_LINES, 0, m_gridVertexCount);

    glBindVertexArray(0);
}

void GLScene::drawWindows(
    const Mat4& vp,
    const std::vector<WindowRender>& windows
) {
    // World-space polygons for the visible windows (the unit quad's corners
    // run through each window's model matrix; uvRect is folded into the
    // corner UVs). A depth slab adds the back face and one wall quad per
    // silhouette segment; the slab's interior is kept SEPARATE from the
    // content face and assembled after all faces -- see the invariant at
    // the assembly below.
    struct SWinSlab {
        SWPoly              front{}; // the content face
        SWPoly              back{};  // the mirrored content face
        std::vector<SWPoly> inner;   // the walls
        float               dist = 0.f;
        bool                fromBehind = false; // eye on the back side
    };
    std::vector<SWinSlab> slabs;
    slabs.reserve(windows.size());

    const Vec3 eye = m_camera.position;

    for (const auto& window : windows) {
        if (!window.texture)
            continue;

        if (window.alpha <= 0.0f)
            continue;

        if (window.width <= 0.0f || window.height <= 0.0f)
            continue;

        const Mat4 model =
            Mat4::translation({window.x, window.y, window.z}) *
            Mat4::rotationY(window.yaw) *
            Mat4::rotationX(window.pitch) *
            Mat4::rotationZ(window.roll) *
            Mat4::scale({window.width, window.height, 1.0f});

        // Local quad point -> world, through the model matrix columns (the
        // scale in m[12..] means this is NOT a plain matrix*vec).
        const auto WORLD = [&model](float lx, float ly, float lz) {
            return Vec3{
                model.m[0] * lx + model.m[4] * ly + model.m[8] * lz + model.m[12],
                model.m[1] * lx + model.m[5] * ly + model.m[9] * lz + model.m[13],
                model.m[2] * lx + model.m[6] * ly + model.m[10] * lz + model.m[14],
            };
        };

        const float CU[4] = {0.f, 1.f, 1.f, 0.f};
        const float CV[4] = {0.f, 0.f, 1.f, 1.f};
        const float LX[4] = {-0.5f, 0.5f, 0.5f, -0.5f};
        const float LY[4] = {-0.5f, -0.5f, 0.5f, 0.5f};

        SWinSlab slab;
        slab.front.tex   = window.texture;
        slab.front.alpha = window.alpha;

        for (int c = 0; c < 4; ++c) {
            SWVert V{
                WORLD(LX[c], LY[c], 0.f),
                window.u0 + (window.u1 - window.u0) * CU[c],
                window.v0 + (window.v1 - window.v0) * CV[c],
            };
            slab.front.verts.push_back(V);
        }

        // Which side of the window the eye is on: the slab's painter order
        // flips with it (see the assembly below).
        const Vec3 NORMAL =
            normalize(Vec3{model.m[8], model.m[9], model.m[10]});
        slab.fromBehind =
            dot(eye - Vec3{window.x, window.y, window.z}, NORMAL) < 0.0f;

        // --- depth slab -----------------------------------------------------
        // Extruded backwards along the window's normal: the front face keeps
        // its exact plane, so picking, input mapping and collision geometry
        // are untouched. Walls hug the captured alpha silhouette (when one
        // was traced), so rounded corners keep their shape, and every wall
        // samples its silhouette texel -- the window texture's edge colors
        // paint the whole side.
        if (window.depth > 0.0f) {
            const float DEPTH = window.depth;

            // Back face: the mirrored content quad at the rear plane. It
            // renders ONLY for an eye on the back side (see the assembly):
            // each content face is visible exclusively from its own side,
            // so the two never stack through a translucent window.
            slab.back.tex   = window.texture;
            slab.back.alpha = window.alpha;

            for (int c = 0; c < 4; ++c) {
                SWVert V{
                    WORLD(LX[c], LY[c], 0.f) - NORMAL * DEPTH,
                    window.u0 + (window.u1 - window.u0) * CU[c],
                    window.v0 + (window.v1 - window.v0) * CV[c],
                };
                slab.back.verts.push_back(V);
            }

            const auto SUBRECT_UV = [&window](const Vec2& P) {
                return Vec2{
                    window.u0 + (window.u1 - window.u0) * P.x,
                    // Outline y is top-down (row 0 = the box's top), and v1
                    // is the subrect's top edge: v runs from v1 (y=0) to
                    // v0 (y=1).
                    window.v1 + (window.v0 - window.v1) * P.y,
                };
            };

            // Wall loops: the traced silhouette when available, else the
            // plain box outline (first frames before a mask was read back).
            //
            // Every segment is emitted TWICE-SIDED -- no facing test: a
            // translucent window must show its FAR walls through the front
            // face (they blend in BSP order, far first), and for opaque
            // windows the hidden walls are simply depth-rejected by the
            // front face drawn after them.
            const auto EMIT_WALLS = [&](const std::vector<Vec2>& pts,
                                        const std::vector<Vec2>& uvs) {
                const size_t N = pts.size();
                if (N < 3)
                    return;

                for (size_t i = 0; i < N; ++i) {
                    const Vec2& A  = pts[i];
                    const Vec2& B  = pts[(i + 1) % N];
                    const Vec2& UA = uvs[i];
                    const Vec2& UB = uvs[(i + 1) % N];

                    const Vec3 AF = WORLD(A.x - 0.5f, 0.5f - A.y, 0.f);
                    const Vec3 BF = WORLD(B.x - 0.5f, 0.5f - B.y, 0.f);

                    const Vec2 SUA = SUBRECT_UV(UA);
                    const Vec2 SUB = SUBRECT_UV(UB);

                    SWPoly wall;
                    wall.tex   = window.texture;
                    wall.alpha = window.alpha;

                    wall.verts.push_back(SWVert{AF, SUA.x, SUA.y});
                    wall.verts.push_back(SWVert{BF, SUB.x, SUB.y});
                    wall.verts.push_back(
                        SWVert{BF - NORMAL * DEPTH, SUB.x, SUB.y});
                    wall.verts.push_back(
                        SWVert{AF - NORMAL * DEPTH, SUA.x, SUA.y});

                    slab.inner.push_back(std::move(wall));
                }
            };

            if (window.outlines && !window.outlines->empty()) {
                for (const auto& LOOP : *window.outlines)
                    EMIT_WALLS(LOOP.pts, LOOP.uvs);
            } else {
                // No silhouette traced (yet) -- the mask readback may fail
                // legitimately. The slab must never lose its sides: fall
                // back to the plain box outline.
                static const std::vector<Vec2> RECT = {
                    {0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
                EMIT_WALLS(RECT, RECT);
            }
        }

        const float DX = slab.front.verts[0].p.x - eye.x;
        const float DY = slab.front.verts[0].p.y - eye.y;
        const float DZ = slab.front.verts[0].p.z - eye.z;
        slab.dist = DX * DX + DY * DY + DZ * DZ;

        slabs.push_back(std::move(slab));
    }

    // Painter-order assembly, windows far-to-near (coplanar overlapping
    // faces blend far first). Within a window the surfaces are ordered
    // far-to-near FOR THE EYE'S SIDE OF THE SLAB -- walls then face from
    // the front; face then walls from behind. This order survives the BSP
    // unchanged for all non-crossing polys (the builder files every
    // behind-or-on-plane poly into the node's coplanar list in insertion
    // order), so whichever surface is farthest blends FIRST and a
    // translucent slab shows its far walls through the near face from BOTH
    // sides -- no ordering luck. Genuinely crossing polys are still split
    // by the node planes as before.
    std::sort(
        slabs.begin(),
        slabs.end(),
        [](const SWinSlab& a, const SWinSlab& b) { return a.dist > b.dist; });

    std::vector<SWPoly> polys;
    polys.reserve(slabs.size() * 8);

    for (auto& S : slabs) {
        // Walls first, then the content face the eye is actually on: the
        // other content face is not emitted at all, so the two never stack
        // through a translucent window -- from the front you see the front
        // face (plus the far walls through it), from behind the back face
        // (plus the far walls through it).
        for (auto& P : S.inner)
            polys.push_back(std::move(P));

        polys.push_back(std::move(S.fromBehind ? S.back : S.front));
    }

    if (polys.empty())
        return;

    // Exact ordering: split crossing quads along each other's planes and
    // traverse back-to-front from the eye.
    SBspNode root;
    bspBuild(root, polys, 0);

    std::vector<SWPoly> ordered;
    bspTraverse(root, eye, [&ordered](const SWPoly& P) {
        ordered.push_back(P);
    });

    if (ordered.empty())
        return;

    if (!m_polyVAO) {
        glGenVertexArrays(1, &m_polyVAO);
        glGenBuffers(1, &m_polyVBO);

        glBindVertexArray(m_polyVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_polyVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    glUseProgram(m_sceneProgram);
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, vp.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTexture, 0);
    // Half a mip level sharper than the GPU picks: mipmapped text reads
    // soft from afar. At 15 m (3.7 window pixels to a screen pixel) -0.5
    // came 0.9 dB closer to an ideal downscale than 0, -1.0 lost the
    // mipmaps' gain at 10 m (measured 2026-10-07, larch-rice sandbox).
    glUniform1f(m_sceneLodBias, -0.5f);

    // Tested against the room's depth, writing none: the BSP order above is
    // already exact among the windows, and depth written by one of them made
    // a coplanar one drawn after it lose to rounding noise -- a desktop wall
    // puts the rice's screen-sized layer in the plane of every window, and
    // the window behind it vanished. The featured window is not tested.
    if (m_windowsOnTop)
        glDisable(GL_DEPTH_TEST);
    else
        glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        GL_ZERO, GL_ONE
    );

    glBindVertexArray(m_polyVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_polyVBO);

    glActiveTexture(GL_TEXTURE0);

    // Consecutive polys with the same texture and alpha merge into ONE
    // upload + draw: a depth slab adds ~100 wall quads per window, and a
    // draw call per wall quad would sink the frame. Only CONSECUTIVE polys
    // group, so the BSP's back-to-front blend order is untouched -- batches
    // interleave exactly where windows overlap.
    std::vector<float> verts;

    for (size_t i = 0; i < ordered.size();) {
        size_t j = i;
        while (j < ordered.size() && ordered[j].tex == ordered[i].tex &&
               ordered[j].alpha == ordered[i].alpha)
            ++j;

        verts.clear();

        for (size_t k = i; k < j; ++k) {
            const auto& P = ordered[k];

            for (size_t v = 1; v + 1 < P.verts.size(); ++v) {
                const auto& A = P.verts[0];
                const auto& B = P.verts[v];
                const auto& C = P.verts[v + 1];

                verts.insert(verts.end(), {
                    A.p.x, A.p.y, A.p.z, A.u, A.v,
                    B.p.x, B.p.y, B.p.z, B.u, B.v,
                    C.p.x, C.p.y, C.p.z, C.u, C.v,
                });
            }
        }

        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
                     verts.data(), GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(0));
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(3 * sizeof(float)));

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, ordered[i].tex);
        glUniform1i(m_sceneTextured, 1);
        glUniform4f(m_sceneColorUniform, 1.f, 1.f, 1.f, ordered[i].alpha);

        glDrawArrays(GL_TRIANGLES, 0,
                     static_cast<GLint>(verts.size() / 5));

        i = j;
    }

    glUniform1f(m_sceneLodBias, 0.0f);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDepthMask(GL_TRUE);
}

// Restored from the pre-unification revision: the black cross with the
// white outline at the screen centre (screen-space, scene program).
void GLScene::drawCrosshair(int width, int height) {
    if (!m_crosshairVAO || !m_crosshairVBO || width <= 0 || height <= 0)
        return;

    const float pxX = 2.0f / static_cast<float>(width);
    const float pxY = 2.0f / static_cast<float>(height);

    // A window in use (setCursorPoint): the cross where its pointer is. The
    // scene's NDC has y up, the target's (Hyprland's framebuffer) y down.
    float atX = 0.0f, atY = 0.0f;
    if (m_cursorOn) {
        const auto& M = m_lastVP.m;
        const float X = M[0] * m_cursorAt.x + M[4] * m_cursorAt.y + M[8] * m_cursorAt.z + M[12];
        const float Y = M[1] * m_cursorAt.x + M[5] * m_cursorAt.y + M[9] * m_cursorAt.z + M[13];
        const float W = M[3] * m_cursorAt.x + M[7] * m_cursorAt.y + M[11] * m_cursorAt.z + M[15];
        if (W > 1e-4f)
            atX = X / W, atY = -Y / W;
    }

    constexpr float OUTER_PX   = 10.0f;
    constexpr float INNER_PX   = 3.5f;
    constexpr float THICK_PX   = 1.4f;
    constexpr float OUTLINE_PX = 1.0f;

    const auto addRect = [&](std::vector<float>& v, float x0, float y0, float x1, float y1) {
        const float z = 0.0f;
        const float u = 0.0f;
        const float t = 0.0f;
        x0 += atX, x1 += atX, y0 += atY, y1 += atY;
        const float verts[] = {
            x0,y0,z,u,t, x1,y0,z,u,t, x1,y1,z,u,t,
            x0,y0,z,u,t, x1,y1,z,u,t, x0,y1,z,u,t,
        };
        v.insert(v.end(), std::begin(verts), std::end(verts));
    };

    // One layer of the crosshair: four arms with the given pixel metrics and
    // color, uploaded and drawn as screen-space geometry.
    const auto drawLayer = [&](float outer, float inner, float thick,
                               float r, float g, float b, float a) {
        const float outerX = outer * pxX;
        const float innerX = inner * pxX;
        const float halfTX = (thick * pxX) * 0.5f;
        const float outerY = outer * pxY;
        const float innerY = inner * pxY;
        const float halfTY = (thick * pxY) * 0.5f;

        std::vector<float> verts;
        verts.reserve(24 * 5);

        // Horizontal arm.
        addRect(verts, -outerX, -halfTY, -innerX, halfTY);
        addRect(verts,  innerX, -halfTY,  outerX, halfTY);
        // Vertical arm.
        addRect(verts, -halfTX,  innerY, halfTX, outerY);
        addRect(verts, -halfTX, -outerY, halfTX, -innerY);

        glUseProgram(m_sceneProgram);

        const Mat4 IDENTITY = Mat4::identity();
        glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, IDENTITY.m.data());
        glUniform1i(m_sceneTextured, 0);
        glUniform4f(m_sceneColorUniform, r, g, b, a);
        glUniform4f(m_sceneUVRect, 0.0f, 0.0f, 1.0f, 1.0f);

        glBindBuffer(GL_ARRAY_BUFFER, m_crosshairVBO);
        glBufferSubData(
            GL_ARRAY_BUFFER, 0,
            static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
            verts.data()
        );

        glBindVertexArray(m_crosshairVAO);
        glDrawArrays(GL_TRIANGLES, 0, 24);
        glBindVertexArray(0);
    };

    // Black core with a white outline: the slightly larger white cross is
    // drawn first, so the black one keeps full contrast on any background.
    // Red while the process gun is out.
    drawLayer(OUTER_PX + OUTLINE_PX, std::max(0.0f, INNER_PX - OUTLINE_PX),
        THICK_PX + 2.0f * OUTLINE_PX, 1.0f, 1.0f, 1.0f, 1.0f);
    if (m_gunOn)
        drawLayer(OUTER_PX, INNER_PX, THICK_PX, 0.9f, 0.12f, 0.1f, 1.0f);
    else
        drawLayer(OUTER_PX, INNER_PX, THICK_PX, 0.0f, 0.0f, 0.0f, 1.0f);

    // The kill being held: a ring filling clockwise from the top.
    if (!m_gunOn || m_gunCharge <= 0.0f)
        return;
    if (!m_ringVAO) {
        glGenVertexArrays(1, &m_ringVAO);
        glGenBuffers(1, &m_ringVBO);
        glBindVertexArray(m_ringVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_ringVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(3 * sizeof(float)));
    }
    constexpr int   SEGMENTS = 48;
    constexpr float RADIUS_PX = 18.0f, RING_PX = 3.0f;
    const int       N = std::max(1, static_cast<int>(std::ceil(SEGMENTS * std::min(m_gunCharge, 1.0f))));
    std::vector<float> ring;
    ring.reserve(N * 30);
    const auto AT = [&](float a, float r) {
        return std::array<float, 2>{std::sin(a) * r * pxX, std::cos(a) * r * pxY};
    };
    for (int i = 0; i < N; ++i) {
        const float A0 = 2.0f * 3.14159265f * std::min(m_gunCharge, 1.0f) * i / N;
        const float A1 = 2.0f * 3.14159265f * std::min(m_gunCharge, 1.0f) * (i + 1) / N;
        const auto  I0 = AT(A0, RADIUS_PX - RING_PX * 0.5f), O0 = AT(A0, RADIUS_PX + RING_PX * 0.5f);
        const auto  I1 = AT(A1, RADIUS_PX - RING_PX * 0.5f), O1 = AT(A1, RADIUS_PX + RING_PX * 0.5f);
        const float Q[] = {I0[0], I0[1], 0.f, 0.f, 0.f, O0[0], O0[1], 0.f, 0.f, 0.f, O1[0], O1[1], 0.f, 0.f, 0.f,
                           I0[0], I0[1], 0.f, 0.f, 0.f, O1[0], O1[1], 0.f, 0.f, 0.f, I1[0], I1[1], 0.f, 0.f, 0.f};
        ring.insert(ring.end(), std::begin(Q), std::end(Q));
    }
    glUseProgram(m_sceneProgram);
    const Mat4 IDENTITY = Mat4::identity();
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, IDENTITY.m.data());
    glUniform1i(m_sceneTextured, 0);
    if (m_gunHung)
        glUniform4f(m_sceneColorUniform, 0.95f, 0.15f, 0.1f, 1.0f);
    else
        glUniform4f(m_sceneColorUniform, 1.0f, 1.0f, 1.0f, 1.0f);
    glUniform4f(m_sceneUVRect, 0.0f, 0.0f, 1.0f, 1.0f);
    glBindVertexArray(m_ringVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_ringVBO);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(ring.size() * sizeof(float)), ring.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, N * 6);
    glBindVertexArray(0);
}

void GLScene::drawFullscreen(
    float alpha
) {
    glUseProgram(m_blitProgram);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_sceneColor);

    glUniform1i(m_blitTexture, 0);
    glUniform1f(m_blitAlpha, alpha);

    glBindVertexArray(m_fullscreenVAO);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindVertexArray(0);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void GLScene::requestProbe() {
    m_probeRequested = true;
}

bool GLScene::probeValid() const {
    return m_probeValid;
}

const unsigned char* GLScene::probeRGBA() const {
    return m_probe;
}

    // F3 HUD text: a streaming quad per set font bit, screen-space ortho.
// font8x8 is public domain (daniel hepper / marcel sondaar).
void GLScene::drawDebugOverlay(int width, int height) {
    if (!m_debugOverlay)
        return;

    // The HUD must land in the scene texture or the composite never picks
    // it up. Save and restore whatever framebuffer was current.
    GLint oldFBO = 0, oldViewport[4] = {0, 0, 0, 0};
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &oldFBO);
    glGetIntegerv(GL_VIEWPORT, oldViewport);

    glBindFramebuffer(GL_FRAMEBUFFER, m_sceneFBO);
    glViewport(0, 0, width, height);

    // The lines: coordinates, view angles, fps, map size.
    const auto& CAM = m_camera;
    char line0[96], line1[96], line2[96], line3[96];
    snprintf(line0, sizeof(line0), "XYZ %.2f %.2f %.2f",
             CAM.position.x, CAM.position.y, CAM.position.z);
    snprintf(line1, sizeof(line1), "YAW %.1f  PIT %.1f",
             CAM.yaw * 180.0f / 3.14159265f, CAM.pitch * 180.0f / 3.14159265f);
    snprintf(line2, sizeof(line2), "FPS %.0f", m_debugFps);
    size_t tris = 0;
    size_t loaded = 0;
    for (const auto& S : m_slots)
        if (S.model && S.model->loaded()) {
            ++loaded;
            tris += S.model->triangles().size();
        }

    snprintf(line3, sizeof(line3), "MAP %zu/%zu objects %zu tris",
             loaded, m_slots.size(), tris);

    const char* LINES[4] = {line0, line1, line2, line3};

    constexpr float GLYPH = 8.0f;
    constexpr float SCALE = 2.0f;   // 16 px tall text
    constexpr float LINE  = GLYPH * SCALE + 4.0f;

    std::vector<float> verts;
    verts.reserve(64 * 1024);

    // pos(3) + uv(2) -- the SCENE program's layout, so the text rides the
    // same shader that already renders the floor and windows. A dedicated
    // mini-program rendered nothing on real GLES; the scene program is the
    // one path guaranteed to work.
    const auto QUAD = [&](float l, float t, float r, float b) {
        verts.insert(verts.end(), {
            l, t, 0, 0, 0,  r, t, 0, 0, 0,  l, b, 0, 0, 0,
            l, b, 0, 0, 0,  r, t, 0, 0, 0,  r, b, 0, 0, 0,
        });
    };

    const auto EMIT_TEXT = [&](float x, float y) {
        for (int li = 0; li < 4; ++li) {
            float cx = x;
            for (const char* P = LINES[li]; *P; ++P) {
                const auto ROWS = font8x8_basic[static_cast<unsigned char>(*P)];
                for (int row = 0; row < 8; ++row) {
                    const unsigned BITS = ROWS[row];
                    if (!BITS)
                        continue;
                    for (int col = 0; col < 8; ++col) {
                        if (!(BITS & (1u << col)))
                            continue;
                        QUAD(cx + col * SCALE, y + li * LINE + row * SCALE,
                             cx + col * SCALE + SCALE,
                             y + li * LINE + row * SCALE + SCALE);
                    }
                }
                cx += GLYPH * SCALE;
            }
        }
    };

    // Shadow text (offset +2,+2) then white -- readable without a bar.
    EMIT_TEXT(10.0f, 12.0f);
    const size_t SHADOW_VERTS = verts.size() / 5;
    EMIT_TEXT(8.0f, 10.0f);

    if (verts.empty())
        return;

    if (!m_textVAO) {
        glGenVertexArrays(1, &m_textVAO);
        glGenBuffers(1, &m_textVBO);
        glBindVertexArray(m_textVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_textVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(3 * sizeof(float)));
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    glBindVertexArray(m_textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_textVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
                 verts.data(), GL_DYNAMIC_DRAW);

    // Pixel coords -> NDC, y down. The negative y scale mirrors winding,
    // so face culling stays off for this pass.
    const Mat4 ORTHO = Mat4::translation(Vec3{-1.f, 1.f, 0.f}) *
        Mat4::scale(Vec3{2.0f / width, -2.0f / height, 1.0f});

    glUseProgram(m_sceneProgram);
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, ORTHO.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTextured, 0);

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glDisable(GL_CULL_FACE);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          reinterpret_cast<void*>(0));
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));

    glUniform4f(m_sceneColorUniform, 0.f, 0.f, 0.f, 0.9f);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLint>(SHADOW_VERTS));

    glUniform4f(m_sceneColorUniform, 1.f, 1.f, 1.f, 1.f);
    glDrawArrays(GL_TRIANGLES, static_cast<GLint>(SHADOW_VERTS),
                 static_cast<GLint>(verts.size() / 5 - SHADOW_VERTS));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(oldFBO));
    glViewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3]);
}
void GLScene::drawRoom(const Mat4& vp, const ViewWindow& view,
                       const std::vector<WindowRender>& windows, float dt,
                       bool primary, bool sight) {
    drawPanorama(view);

    for (auto& S : m_slots) {
        if (!S.model || !S.model->loaded())
            continue;

        S.model->draw(vp, m_camera.position);

        // Red x-ray wireframe of the collision triangles (debug).
        if (!sight)
            S.model->drawDebug(vp);
    }

    if (m_pDbgOn && !sight)
        drawCapsule(vp, m_pDbgCenter, nullptr, Vec3{0.2f, 1.0f, 0.3f}, false);
    // Arch blue, the default accent of Larch's palette.
    if (m_compOn && !sight)
        drawCapsule(vp, m_compCenter, &m_compYaw, Vec3{0.09f, 0.576f, 0.82f}, true);
    if ((m_playerVisible || sight) && m_player.loaded()) {
        m_player.setPose(m_playerFeet, m_playerYaw, m_playerCfg.scale,
                         m_playerCfg.posOffset + m_playerCfg.centerOffset,
                         m_playerCfg.rotDeg);
        m_player.setFlat(m_playerCfg.flat);
        m_player.setEmissiveScale(m_playerCfg.emissiveScale);
        if (primary)
            m_player.update(dt);
        m_player.draw(vp, m_camera.position);
    } else if (sight) {
        // The player without a model: orange, apart from the companion's
        // blue and the Moon's grey. The capsule's centre is set every frame.
        drawCapsule(vp, m_pDbgCenter, nullptr, Vec3{1.0f, 0.55f, 0.1f}, true);
    }

    drawPortals(vp);

    if (m_gridVisible) {
        drawFloor(vp);

        glEnable(GL_BLEND);
        glBlendFuncSeparate(
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            GL_ZERO, GL_ONE
        );
        glDepthMask(GL_FALSE);

        drawGrid(vp);
    }

    drawShadows(vp);

    // Windows last, back to front in the BSP's exact order: crossing quads
    // are split along each other, and translucency composites in order. A
    // featured window (F2, F4) comes after the rest and the dark over them --
    // for the player's eyes: the companion's eye sees the room as it stands.
    if (!m_featuredId || sight)
        drawWindows(vp, windows);
    else {
        std::vector<WindowRender> rest, featured;
        for (const auto& W : windows)
            (W.id == m_featuredId ? featured : rest).push_back(W);
        drawWindows(vp, rest);
        if (m_featuredDim > 0.0f)
            drawDim(m_featuredDim);
        m_windowsOnTop = true;
        drawWindows(vp, featured);
        m_windowsOnTop = false;
    }

    glDepthMask(GL_TRUE);
}

void GLScene::setPortals(const std::vector<SPortalSpec>& portals) {
    std::vector<SPortalGL> next;
    next.reserve(portals.size());
    for (const auto& SPEC : portals) {
        SPortalGL P;
        P.spec = SPEC;
        // The same picture keeps its texture.
        const auto OLD = std::find_if(m_portals.begin(), m_portals.end(),
                                      [&](const SPortalGL& o) { return o.spec.name == SPEC.name; });
        if (OLD != m_portals.end() && OLD->loaded == SPEC.image) {
            P.tex    = std::exchange(OLD->tex, 0u);
            P.aspect = OLD->aspect;
            P.loaded = OLD->loaded;
        }
        next.push_back(std::move(P));
    }
    for (auto& OLD : m_portals)
        if (OLD.tex)
            m_portalTrash.push_back(OLD.tex);
    m_portals = std::move(next);
}

float GLScene::portalAspect(const std::string& name) const {
    for (const auto& P : m_portals)
        if (P.spec.name == name)
            return P.aspect;
    return 0.0f;
}

void GLScene::refreshPortals() {
    if (!m_portalTrash.empty()) {
        glDeleteTextures(static_cast<GLsizei>(m_portalTrash.size()), m_portalTrash.data());
        m_portalTrash.clear();
    }
    for (auto& P : m_portals) {
        if (P.loaded == P.spec.image)
            continue;
        P.loaded = P.spec.image; // one attempt per path, failed or not
        Hyprgraphics::CImage image(P.spec.image);
        auto surface = image.success() ? image.cairoSurface() : nullptr;
        if (!surface || surface->status() != CAIRO_STATUS_SUCCESS)
            continue;
        const int W = static_cast<int>(surface->size().x), H = static_cast<int>(surface->size().y);
        const int STRIDE = surface->stride();
        const auto* SRC  = surface->data();
        if (W <= 0 || H <= 0 || !SRC)
            continue;

        // Cairo ARGB32 is premultiplied BGRA; the scene blends straight alpha.
        std::vector<unsigned char> px(static_cast<size_t>(W) * H * 4);
        for (int y = 0; y < H; ++y) {
            const auto* row = SRC + static_cast<size_t>(y) * STRIDE;
            auto* dst = px.data() + static_cast<size_t>(y) * W * 4;
            for (int x = 0; x < W; ++x) {
                const unsigned A = row[x * 4 + 3];
                const auto UN = [A](unsigned c) { return A ? static_cast<unsigned char>(std::min(255u, c * 255u / A)) : 0; };
                dst[x * 4 + 0] = UN(row[x * 4 + 2]);
                dst[x * 4 + 1] = UN(row[x * 4 + 1]);
                dst[x * 4 + 2] = UN(row[x * 4 + 0]);
                dst[x * 4 + 3] = static_cast<unsigned char>(A);
            }
        }
        if (P.tex)
            glDeleteTextures(1, &P.tex);
        glGenTextures(1, &P.tex);
        glBindTexture(GL_TEXTURE_2D, P.tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        P.aspect = static_cast<float>(H) / static_cast<float>(W);
    }
}

void GLScene::drawPortals(const Mat4& vp) {
    if (m_portals.empty())
        return;
    if (!m_portalVAO) {
        glGenVertexArrays(1, &m_portalVAO);
        glGenBuffers(1, &m_portalVBO);
        glBindVertexArray(m_portalVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_portalVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(3 * sizeof(float)));
    }

    glUseProgram(m_sceneProgram);
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, vp.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTexture, 0);
    glUniform1i(m_sceneTextured, 1);
    glUniform4f(m_sceneColorUniform, 1.f, 1.f, 1.f, 1.f);
    // Its glow fades out softly, but only what is clearly there hides what
    // stands behind: the cut writes depth for the opaque part alone.
    glUniform1f(m_sceneAlphaCut, 0.35f);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glBindVertexArray(m_portalVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_portalVBO);
    glActiveTexture(GL_TEXTURE0);

    for (const auto& P : m_portals) {
        if (!P.tex || P.aspect <= 0.0f)
            continue;
        const float W = P.spec.width, H = P.spec.width * P.aspect;
        const Vec3  FACE{std::sin(P.spec.yaw), 0.f, -std::cos(P.spec.yaw)};
        // Seen from in front, looking along -FACE, the picture's left is on
        // the viewer's left.
        const Vec3 R  = normalize(cross(FACE * -1.0f, Vec3{0.f, 1.f, 0.f})) * (W * 0.5f);
        const Vec3 C  = P.spec.base + Vec3{0.f, H * 0.5f, 0.f};
        const Vec3 UP{0.f, H * 0.5f, 0.f};
        const Vec3 TL = C - R + UP, TR = C + R + UP, BR = C + R - UP, BL = C - R - UP;
        const float V[] = {
            TL.x, TL.y, TL.z, 0.f, 0.f,  TR.x, TR.y, TR.z, 1.f, 0.f,  BR.x, BR.y, BR.z, 1.f, 1.f,
            TL.x, TL.y, TL.z, 0.f, 0.f,  BR.x, BR.y, BR.z, 1.f, 1.f,  BL.x, BL.y, BL.z, 0.f, 1.f,
        };
        glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_DYNAMIC_DRAW);
        glBindTexture(GL_TEXTURE_2D, P.tex);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    glUniform1f(m_sceneAlphaCut, 0.0f);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLScene::drawHud(float alpha) {
    if (m_hud.empty())
        return;
    if (!m_hudVAO) {
        glGenVertexArrays(1, &m_hudVAO);
        glGenBuffers(1, &m_hudVBO);
        glBindVertexArray(m_hudVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_hudVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(3 * sizeof(float)));
    }
    glUseProgram(m_sceneProgram);
    const Mat4 I = Mat4::identity();
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, I.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTexture, 0);
    glUniform1i(m_sceneTextured, 1);
    glUniform4f(m_sceneColorUniform, 1.f, 1.f, 1.f, alpha);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(m_hudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_hudVBO);
    for (const auto& Q : m_hud) {
        if (!Q.texture)
            continue;
        const float V[] = {Q.x0, Q.y0, 0.f, Q.u0, Q.v0, Q.x1, Q.y0, 0.f, Q.u1, Q.v0, Q.x1, Q.y1, 0.f, Q.u1, Q.v1,
                           Q.x0, Q.y0, 0.f, Q.u0, Q.v0, Q.x1, Q.y1, 0.f, Q.u1, Q.v1, Q.x0, Q.y1, 0.f, Q.u0, Q.v1};
        glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_DYNAMIC_DRAW);
        glBindTexture(GL_TEXTURE_2D, Q.texture);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLScene::drawShadows(const Mat4& vp) {
    if (m_shadows.empty())
        return;
    if (!m_shadowTex) {
        // The blob: black, its alpha a Gaussian that reaches 0 at the edge
        // of the square, so the quad's border never shows.
        static constexpr int N = 64;
        std::vector<unsigned char> px(N * N * 4, 0);
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                const float U = (x + 0.5f) / N * 2.f - 1.f, V = (y + 0.5f) / N * 2.f - 1.f;
                const float R2 = U * U + V * V;
                const float A = std::exp(-R2 / (2.f * 0.35f * 0.35f)) * std::max(0.f, 1.f - R2);
                px[(y * N + x) * 4 + 3] = static_cast<unsigned char>(std::lround(255.f * A));
            }
        glGenTextures(1, &m_shadowTex);
        glBindTexture(GL_TEXTURE_2D, m_shadowTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    if (!m_shadowVAO) {
        glGenVertexArrays(1, &m_shadowVAO);
        glGenBuffers(1, &m_shadowVBO);
        glBindVertexArray(m_shadowVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_shadowVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(3 * sizeof(float)));
    }

    std::vector<float> V;
    V.reserve(m_shadows.size() * 30);
    for (const auto& S : m_shadows) {
        const Vec3 R = S.right * S.halfW;
        const Vec3 D = Vec3{-S.right.z, 0.f, S.right.x} * S.halfD;
        const Vec3 A = S.center - R - D, B = S.center + R - D, C = S.center + R + D, E = S.center - R + D;
        const float Q[] = {A.x, A.y, A.z, 0.f, 0.f, B.x, B.y, B.z, 1.f, 0.f, C.x, C.y, C.z, 1.f, 1.f,
                           A.x, A.y, A.z, 0.f, 0.f, C.x, C.y, C.z, 1.f, 1.f, E.x, E.y, E.z, 0.f, 1.f};
        V.insert(V.end(), std::begin(Q), std::end(Q));
    }

    glUseProgram(m_sceneProgram);
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, vp.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTexture, 0);
    glUniform1i(m_sceneTextured, 1);
    // On the floor, under everything that stands on it: tested against the
    // depth, never written. main lifts it a centimetre off the floor; the
    // offset here is constant, never by slope -- seen at a grazing angle a
    // sloped offset pulled it in front of the deck's lit edge.
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(0.f, -4.f);
    const GLboolean CULL = glIsEnabled(GL_CULL_FACE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_shadowTex);
    glBindVertexArray(m_shadowVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_shadowVBO);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(V.size() * sizeof(float)), V.data(), GL_DYNAMIC_DRAW);
    for (size_t i = 0; i < m_shadows.size(); ++i) {
        glUniform4f(m_sceneColorUniform, 1.f, 1.f, 1.f, m_shadows[i].alpha);
        glDrawArrays(GL_TRIANGLES, static_cast<GLint>(i * 6), 6);
    }
    glDisable(GL_POLYGON_OFFSET_FILL);
    if (CULL)
        glEnable(GL_CULL_FACE);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLScene::drawDim(float dim) {
    if (!m_dimVAO) {
        // Two triangles over the whole target, in clip space.
        static constexpr float QUAD[] = {
            -1.f, -1.f, 0.f, 0.f, 0.f,  1.f, -1.f, 0.f, 0.f, 0.f,  1.f, 1.f, 0.f, 0.f, 0.f,
            -1.f, -1.f, 0.f, 0.f, 0.f,  1.f,  1.f, 0.f, 0.f, 0.f, -1.f, 1.f, 0.f, 0.f, 0.f,
        };
        glGenVertexArrays(1, &m_dimVAO);
        glGenBuffers(1, &m_dimVBO);
        glBindVertexArray(m_dimVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_dimVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD), QUAD, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                              reinterpret_cast<void*>(3 * sizeof(float)));
    }

    glUseProgram(m_sceneProgram);
    const Mat4 I = Mat4::identity();
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, I.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTextured, 0);
    glUniform4f(m_sceneColorUniform, 0.f, 0.f, 0.f, std::clamp(dim, 0.0f, 1.0f));

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glBindVertexArray(m_dimVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLScene::drawSight(const std::vector<WindowRender>& windows) {
    // One picture on its way back at a time; this one waits for it.
    if (m_sightFence)
        return;
    m_sightWanted = false;

    constexpr GLsizeiptr BYTES = GLsizeiptr{kSightWidth} * kSightHeight * 4;
    if (!m_sightFBO) {
        glGenFramebuffers(1, &m_sightFBO);
        glGenRenderbuffers(1, &m_sightColor);
        glGenRenderbuffers(1, &m_sightDepth);
        glGenBuffers(1, &m_sightPBO);

        glBindRenderbuffer(GL_RENDERBUFFER, m_sightColor);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, kSightWidth, kSightHeight);
        glBindRenderbuffer(GL_RENDERBUFFER, m_sightDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, kSightWidth, kSightHeight);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, m_sightFBO);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_sightColor);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_sightDepth);

        GLint oldPack = 0;
        glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &oldPack);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, m_sightPBO);
        glBufferData(GL_PIXEL_PACK_BUFFER, BYTES, nullptr, GL_STREAM_READ);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(oldPack));
    }

    glBindFramebuffer(GL_FRAMEBUFFER, m_sightFBO);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        sightFailed("The eye's framebuffer is incomplete");
        return;
    }

    glViewport(0, 0, kSightWidth, kSightHeight);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glClearColor(0.012f, 0.019f, 0.032f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 90 degrees across: the view plane at distance 1 spans -1..1.
    constexpr float HALF_H = static_cast<float>(kSightHeight) / kSightWidth;
    const ViewWindow VIEW{-1.0f, 1.0f, -HALF_H, HALF_H};
    constexpr float ZNEAR = 0.05f, ZFAR = 200.0f;
    const Mat4 projection = Mat4::frustum(VIEW.left * ZNEAR, VIEW.right * ZNEAR,
                                          VIEW.bottom * ZNEAR, VIEW.top * ZNEAR, ZNEAR, ZFAR);

    // The panorama and the shading read m_camera: the eye stands in for it.
    std::swap(m_camera, m_sightCam);
    drawRoom(projection * m_camera.view(), VIEW, windows, 0.0f, false, true);
    std::swap(m_camera, m_sightCam);

    GLint oldPack = 0;
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &oldPack);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, m_sightPBO);
    glReadPixels(0, 0, kSightWidth, kSightHeight, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(oldPack));
    m_sightFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!m_sightFence)
        sightFailed("glFenceSync failed");
}

void GLScene::readSight() {
    if (!m_sightFence)
        return;

    const auto FENCE = static_cast<GLsync>(m_sightFence);
    const GLenum STATE = glClientWaitSync(FENCE, 0, 0);
    if (STATE == GL_TIMEOUT_EXPIRED)
        return;
    glDeleteSync(FENCE);
    m_sightFence = nullptr;

    if (m_sightDrop) {
        m_sightDrop = false;
        return;
    }
    if (STATE == GL_WAIT_FAILED) {
        sightFailed("glClientWaitSync failed");
        return;
    }

    constexpr GLsizeiptr BYTES = GLsizeiptr{kSightWidth} * kSightHeight * 4;
    GLint oldPack = 0;
    glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &oldPack);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, m_sightPBO);
    const auto* MAPPED =
        static_cast<const unsigned char*>(glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, BYTES, GL_MAP_READ_BIT));
    if (MAPPED) {
        m_sightPixels.assign(MAPPED, MAPPED + BYTES);
        m_sightResult = ESight::Picture;
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    } else
        sightFailed("glMapBufferRange failed");
    glBindBuffer(GL_PIXEL_PACK_BUFFER, static_cast<GLuint>(oldPack));
}

bool GLScene::render(
    unsigned int targetFBO,
    int width,
    int height,
    float alpha,
    float dt,
    const std::vector<WindowRender>& windows,
    const ViewWindow* view,
    bool primary
) {
    if (width <= 0 || height <= 0)
        return false;

    if (!initialize())
        return false;

    if (!ensureSceneFramebuffer(width, height))
        return false;

    // Once per frame, on the monitor the view is built for: the others draw
    // the same instant, they do not advance it.
    if (primary)
        m_time += std::clamp(dt, 0.0f, 0.1f);

    GLint oldDrawFBO = 0;
    GLint oldReadFBO = 0;
    GLint oldProgram = 0;
    GLint oldVAO = 0;
    GLint oldArrayBuffer = 0;
    GLint oldActiveTexture = 0;
    GLint oldTexture0 = 0;
    GLint oldViewport[4] = {};

    GLint oldBlendSrcRGB = 0;
    GLint oldBlendDstRGB = 0;
    GLint oldBlendSrcAlpha = 0;
    GLint oldBlendDstAlpha = 0;
    GLint oldBlendEqRGB = 0;
    GLint oldBlendEqAlpha = 0;

    GLint oldDepthFunc = 0;
    GLfloat oldLineWidth = 1.0f;

    GLboolean oldBlend = GL_FALSE;
    GLboolean oldDepthTest = GL_FALSE;
    GLboolean oldScissor = GL_FALSE;
    GLboolean oldCull = GL_FALSE;
    GLboolean oldDepthMask = GL_TRUE;

    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &oldDrawFBO);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &oldReadFBO);
    glGetIntegerv(GL_CURRENT_PROGRAM, &oldProgram);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &oldVAO);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldArrayBuffer);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &oldActiveTexture);

    glActiveTexture(GL_TEXTURE0);

    glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTexture0);
    glGetIntegerv(GL_VIEWPORT, oldViewport);
    glGetIntegerv(GL_BLEND_SRC_RGB, &oldBlendSrcRGB);
    glGetIntegerv(GL_BLEND_DST_RGB, &oldBlendDstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &oldBlendSrcAlpha);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &oldBlendDstAlpha);
    glGetIntegerv(GL_BLEND_EQUATION_RGB, &oldBlendEqRGB);
    glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &oldBlendEqAlpha);
    glGetIntegerv(GL_DEPTH_FUNC, &oldDepthFunc);
    glGetFloatv(GL_LINE_WIDTH, &oldLineWidth);
    glGetBooleanv(GL_BLEND, &oldBlend);
    glGetBooleanv(GL_DEPTH_TEST, &oldDepthTest);
    glGetBooleanv(GL_SCISSOR_TEST, &oldScissor);
    glGetBooleanv(GL_CULL_FACE, &oldCull);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &oldDepthMask);

    // --- scene pass: rendered offscreen, then composited over the desktop ---
    //
    // The desktop underneath is left exactly as Hyprland drew it and simply
    // fades out via `alpha`. It is never copied or re-projected, so the
    // workspace never appears as an object inside the 3D world.

    glBindFramebuffer(GL_FRAMEBUFFER, m_sceneFBO);

    glViewport(0, 0, width, height);

    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);

    glClearColor(0.012f, 0.019f, 0.032f, 1.0f);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const float aspect =
        static_cast<float>(width) / static_cast<float>(height);

    // The rectangle of the view plane this framebuffer shows, narrowed by the
    // zoom (C key) about the view axis like the fov: with no `view` the
    // symmetric fov, which is exactly the perspective() this replaced.
    const float TANFOV = std::tan(kFovDeg * PI / 360.0f);
    ViewWindow VIEW = view ? *view : ViewWindow{-TANFOV * aspect, TANFOV * aspect, -TANFOV, TANFOV};
    const float ZOOM = std::max(m_zoom, 0.01f);
    VIEW.left /= ZOOM;
    VIEW.right /= ZOOM;
    VIEW.bottom /= ZOOM;
    VIEW.top /= ZOOM;

    constexpr float ZNEAR = 0.05f, ZFAR = 200.0f;
    const Mat4 projection = Mat4::frustum(VIEW.left * ZNEAR, VIEW.right * ZNEAR,
                                          VIEW.bottom * ZNEAR, VIEW.top * ZNEAR, ZNEAR, ZFAR);

    // What picking and the HUD measure against: the primary monitor.
    if (primary) {
        m_width  = width;
        m_height = height;
    }

    const Mat4 vp = projection * m_camera.view();

    // Background panorama first, then the map (it replaces the flat floor
    // when loaded; depth rejects whatever is behind its geometry), then the
    // ground so windows behind it are depth-rejected.
    if (primary) {
        refreshPanorama();
        refreshScene();
        refreshPortals();
    }
    // The player's character (hidden in first person): animated inside
    // render -- the clock and the vertex upload need the EGL context.
    if (primary)
        refreshPlayer();
    drawRoom(vp, VIEW, windows, dt, primary, false);
    if (primary)
        m_lastVP = vp;

    // F3 HUD: always on top of the scene, never part of the 3D pass state.
    if (primary)
        drawDebugOverlay(width, height);

    // Read back one pixel of the offscreen scene while it is still bound. A
    // floor point in the lower half of the screen, where ground and sky are
    // both opaque by construction: alpha below 255 here means the composite
    // cannot become fully opaque no matter what the fade is set to.
    if (primary && m_probeRequested) {
        glReadPixels(
            width / 2,
            height / 4,
            1,
            1,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            m_probe
        );

        m_probeValid    = true;
        m_probeRequested = false;
    }

    // The companion's eye: a picture an earlier frame drew, once the GPU is
    // through with it, then the one asked for now. After the probe, which
    // reads the screen's target.
    if (primary) {
        readSight();
        if (m_sightWanted)
            drawSight(windows);
    }

    // --- composite over whatever Hyprland already drew for this frame ---
    glBindFramebuffer(GL_FRAMEBUFFER, targetFBO);

    glViewport(0, 0, width, height);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_BLEND);

    glBlendFuncSeparate(
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_ONE,
        GL_ONE_MINUS_SRC_ALPHA
    );

    drawFullscreen(std::clamp(alpha, 0.0f, 1.0f));
    drawHud(std::clamp(alpha, 0.0f, 1.0f));
    if (primary)
        drawCrosshair(width, height);

    // --- restore compositor state ---
    glUseProgram(static_cast<GLuint>(oldProgram));
    glBindVertexArray(static_cast<GLuint>(oldVAO));

    glBindBuffer(
        GL_ARRAY_BUFFER,
        static_cast<GLuint>(oldArrayBuffer)
    );

    glActiveTexture(GL_TEXTURE0);

    glBindTexture(
        GL_TEXTURE_2D,
        static_cast<GLuint>(oldTexture0)
    );

    glActiveTexture(static_cast<GLenum>(oldActiveTexture));

    glBlendFuncSeparate(
        oldBlendSrcRGB,
        oldBlendDstRGB,
        oldBlendSrcAlpha,
        oldBlendDstAlpha
    );

    glBlendEquationSeparate(oldBlendEqRGB, oldBlendEqAlpha);
    glDepthFunc(oldDepthFunc);
    glDepthMask(oldDepthMask);
    glLineWidth(oldLineWidth);

    if (oldBlend)
        glEnable(GL_BLEND);
    else
        glDisable(GL_BLEND);

    if (oldDepthTest)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);

    if (oldScissor)
        glEnable(GL_SCISSOR_TEST);
    else
        glDisable(GL_SCISSOR_TEST);

    if (oldCull)
        glEnable(GL_CULL_FACE);
    else
        glDisable(GL_CULL_FACE);

    glViewport(
        oldViewport[0],
        oldViewport[1],
        oldViewport[2],
        oldViewport[3]
    );

    glBindFramebuffer(
        GL_DRAW_FRAMEBUFFER,
        static_cast<GLuint>(oldDrawFBO)
    );

    glBindFramebuffer(
        GL_READ_FRAMEBUFFER,
        static_cast<GLuint>(oldReadFBO)
    );

    return true;
}

void GLScene::setPanoramaPath(const std::string& path) {
    m_panoramaPath = path;
}


void GLScene::mouseMove(
    float dx,
    float dy
) {
    m_camera.mouseDelta(dx, dy);
}

void GLScene::rotateView(
    float yawDelta,
    float pitchDelta
) {
    m_camera.applyLook(yawDelta, pitchDelta);
}

std::uintptr_t GLScene::pick(
    const std::vector<WindowRender>& windows
) const {
    std::vector<RayQuad> quads;
    quads.reserve(windows.size());

    for (const auto& window : windows) {
        RayQuad quad;
        quad.id     = window.id;
        quad.center = {window.x, window.y, window.z};
        quad.width  = window.width;
        quad.height = window.height;
        quad.yaw    = window.yaw;
        quad.pitch  = window.pitch;

        // Match the draw rules exactly: nothing invisible is aimable, so
        // focus can never land on a window you cannot see.
        quad.pickable =
            window.texture != 0 &&
            window.alpha > 0.0f &&
            window.width > 0.0f &&
            window.height > 0.0f;

        quads.emplace_back(quad);
    }

    const RayHit hit =
        rayPickQuads(m_camera.position, m_camera.centerRay(), quads);

    return hit.hit ? hit.id : 0;
}

void GLScene::reset() {
    m_camera = Camera{};
    m_camera.position = {0.0f, 1.5f, 11.0f};
    m_camera.yaw = 0.0f;
    m_camera.pitch = -0.20f;
    m_time = 0.0f;
}

void GLScene::shutdown() {
    for (auto& S : m_slots)
        if (S.model)
            S.model->destroy();
    m_slots.clear();
    destroyGLObjects();
}

void GLScene::destroyGLObjects() {
    for (auto& P : m_portals)
        if (P.tex) {
            glDeleteTextures(1, &P.tex);
            P.tex = 0;
            P.loaded.clear();
        }
    if (!m_portalTrash.empty()) {
        glDeleteTextures(static_cast<GLsizei>(m_portalTrash.size()), m_portalTrash.data());
        m_portalTrash.clear();
    }
    if (m_portalVAO) {
        glDeleteVertexArrays(1, &m_portalVAO);
        glDeleteBuffers(1, &m_portalVBO);
        m_portalVAO = m_portalVBO = 0;
    }
    if (m_dimVAO) {
        glDeleteVertexArrays(1, &m_dimVAO);
        glDeleteBuffers(1, &m_dimVBO);
        m_dimVAO = m_dimVBO = 0;
    }
    if (m_shadowTex) {
        glDeleteTextures(1, &m_shadowTex);
        m_shadowTex = 0;
    }
    if (m_hudVAO) {
        glDeleteVertexArrays(1, &m_hudVAO);
        glDeleteBuffers(1, &m_hudVBO);
        m_hudVAO = m_hudVBO = 0;
    }
    if (m_ringVAO) {
        glDeleteVertexArrays(1, &m_ringVAO);
        glDeleteBuffers(1, &m_ringVBO);
        m_ringVAO = m_ringVBO = 0;
    }
    if (m_shadowVAO) {
        glDeleteVertexArrays(1, &m_shadowVAO);
        glDeleteBuffers(1, &m_shadowVBO);
        m_shadowVAO = m_shadowVBO = 0;
    }
    if (m_sightFence) {
        glDeleteSync(static_cast<GLsync>(m_sightFence));
        m_sightFence = nullptr;
    }
    m_sightDrop = false;
    if (m_sightPBO) {
        glDeleteBuffers(1, &m_sightPBO);
        m_sightPBO = 0;
    }
    if (m_sightFBO) {
        glDeleteRenderbuffers(1, &m_sightDepth);
        glDeleteRenderbuffers(1, &m_sightColor);
        glDeleteFramebuffers(1, &m_sightFBO);
        m_sightFBO = m_sightColor = m_sightDepth = 0;
    }
    if (m_textVBO) {
        glDeleteBuffers(1, &m_textVBO);
        m_textVBO = 0;
    }
    if (m_textVAO) {
        glDeleteVertexArrays(1, &m_textVAO);
        m_textVAO = 0;
    }
    if (m_crosshairVBO) {
        glDeleteBuffers(1, &m_crosshairVBO);
        m_crosshairVBO = 0;
    }

    if (m_crosshairVAO) {
        glDeleteVertexArrays(1, &m_crosshairVAO);
        m_crosshairVAO = 0;
    }

    if (m_fullscreenVBO) {
        glDeleteBuffers(1, &m_fullscreenVBO);
        m_fullscreenVBO = 0;
    }

    if (m_fullscreenVAO) {
        glDeleteVertexArrays(1, &m_fullscreenVAO);
        m_fullscreenVAO = 0;
    }

    if (m_gridVBO) {
        glDeleteBuffers(1, &m_gridVBO);
        m_gridVBO = 0;
    }

    if (m_gridVAO) {
        glDeleteVertexArrays(1, &m_gridVAO);
        m_gridVAO = 0;
    }

    if (m_floorVBO) {
        glDeleteBuffers(1, &m_floorVBO);
        m_floorVBO = 0;
    }

    if (m_floorVAO) {
        glDeleteVertexArrays(1, &m_floorVAO);
        m_floorVAO = 0;
    }

    if (m_quadVBO) {
        glDeleteBuffers(1, &m_quadVBO);
        m_quadVBO = 0;
    }

    if (m_quadVAO) {
        glDeleteVertexArrays(1, &m_quadVAO);
        m_quadVAO = 0;
    }

    if (m_sceneDepth) {
        glDeleteRenderbuffers(1, &m_sceneDepth);
        m_sceneDepth = 0;
    }

    if (m_sceneColor) {
        glDeleteTextures(1, &m_sceneColor);
        m_sceneColor = 0;
    }

    if (m_sceneFBO) {
        glDeleteFramebuffers(1, &m_sceneFBO);
        m_sceneFBO = 0;
    }

    for (auto& T : m_otherSceneTargets) {
        glDeleteRenderbuffers(1, &T.depth);
        glDeleteTextures(1, &T.color);
        glDeleteFramebuffers(1, &T.fbo);
    }
    m_otherSceneTargets.clear();
    m_sceneWidth  = 0;
    m_sceneHeight = 0;

    if (m_sceneProgram) {
        glDeleteProgram(m_sceneProgram);
        m_sceneProgram = 0;
    }

    if (m_blitProgram) {
        glDeleteProgram(m_blitProgram);
        m_blitProgram = 0;
    }

    if (m_panoramaProgram) {
        glDeleteProgram(m_panoramaProgram);
        m_panoramaProgram = 0;
    }



    if (m_panoramaTex) {
        glDeleteTextures(1, &m_panoramaTex);
        m_panoramaTex = 0;
    }

    m_panoramaLoaded.clear();
    m_panoramaMtimeValid = false;

    m_initialized = false;

    m_sceneWidth = 0;
    m_sceneHeight = 0;
}

} // namespace H3D
