#include "GLScene.hpp"


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

        if (window.burn >= 0.f) {
            m_gun.drawPaper(vp, {window.texture,{window.x,window.y,window.z},
                window.width,window.height,window.yaw,window.pitch,window.roll,
                window.u0,window.v0,window.u1,window.v1,window.burn,window.burnOrigin});
            continue;
        }

        // Crumpled: an opaque creased mesh, written to depth before the
        // translucent windows are blended over and behind it.
        if (window.crumple > 0.001f) {
            drawCrumpled(vp, window);
            continue;
        }

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
// --- a window crumpled like paper ---------------------------------------------
// Larch's motion draft (2026-10-08): a window carried to the waste bin
// crumples with its distance to it and goes in as a paper ball. Done the way
// games crumple paper -- a blend between the flat sheet and a crumpled target
// shape, not a sheet simulation: the window's quad as a grid that wrinkles
// first and then gathers into a lumpy ball, rebuilt on the CPU each frame for
// the one window being thrown. Each facet is shaded by how it faces the eye,
// so the creases show.
static float crumpleHash(int x, int y, int z) {
    uint32_t h = static_cast<uint32_t>(x) * 374761393u + static_cast<uint32_t>(y) * 668265263u +
        static_cast<uint32_t>(z) * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return static_cast<float>((h ^ (h >> 16)) & 0xffffu) / 65535.0f * 2.0f - 1.0f;
}

static float crumpleNoise(float x, float y, float z) {
    const float FX = std::floor(x), FY = std::floor(y), FZ = std::floor(z);
    const int   X = static_cast<int>(FX), Y = static_cast<int>(FY), Z = static_cast<int>(FZ);
    const float TX = x - FX, TY = y - FY, TZ = z - FZ;
    const float UX = TX * TX * (3.f - 2.f * TX), UY = TY * TY * (3.f - 2.f * TY), UZ = TZ * TZ * (3.f - 2.f * TZ);
    const auto  LERP = [](float a, float b, float t) { return a + (b - a) * t; };
    const float X00 = LERP(crumpleHash(X, Y, Z), crumpleHash(X + 1, Y, Z), UX);
    const float X10 = LERP(crumpleHash(X, Y + 1, Z), crumpleHash(X + 1, Y + 1, Z), UX);
    const float X01 = LERP(crumpleHash(X, Y, Z + 1), crumpleHash(X + 1, Y, Z + 1), UX);
    const float X11 = LERP(crumpleHash(X, Y + 1, Z + 1), crumpleHash(X + 1, Y + 1, Z + 1), UX);
    return LERP(LERP(X00, X10, UY), LERP(X01, X11, UY), UZ);
}

// Position + UV, the scene program's layout: the crumpled mesh and the bin's
// halo share one buffer.
void GLScene::ensureCrumpleBuffers() {
    if (m_crumpleVAO)
        return;
    glGenVertexArrays(1, &m_crumpleVAO);
    glGenBuffers(1, &m_crumpleVBO);
    glBindVertexArray(m_crumpleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_crumpleVBO);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLScene::drawCrumpled(const Mat4& vp, const WindowRender& w) {
    constexpr int   NX = 28, NY = 18;
    constexpr float PI = 3.14159265f;
    const float E    = std::clamp(w.crumple, 0.0f, 1.0f);
    const float K    = E * E * (3.0f - 2.0f * E);          // flat -> ball
    const float W    = w.width, H = w.height;
    // The ball: a third of the window's size, at most 0.12 m across the
    // middle -- big enough to read as the window, small enough to drop into
    // the bin's 0.58 m opening (a 0.25 m one covered the bin, measured).
    const float R    = std::min(0.17f * std::sqrt(W * H), 0.12f);
    const float FOLD = 0.07f * std::min(W, H) * 4.0f * E * (1.0f - E); // wrinkles, most half way
    const float SEED = static_cast<float>(w.id % 977u) * 0.37f;         // each window its own way

    // Tumbling as it crumples: about the up axis by the spin, a little over.
    const Mat4 ROT = Mat4::rotationY(w.yaw) * Mat4::rotationX(w.pitch) * Mat4::rotationZ(w.roll) *
        Mat4::rotationY(w.spin) * Mat4::rotationX(0.6f * w.spin);
    const Vec3 CENTER{w.x, w.y, w.z};

    std::vector<Vec3> P(static_cast<size_t>((NX + 1) * (NY + 1)));
    for (int j = 0; j <= NY; ++j)
        for (int i = 0; i <= NX; ++i) {
            const float S = static_cast<float>(i) / NX, T = static_cast<float>(j) / NY;
            Vec3 flat{(S - 0.5f) * W, (T - 0.5f) * H, 0.0f};
            flat.z += FOLD * crumpleNoise(S * 5.0f + SEED, T * 5.0f * H / std::max(W, 1e-3f), 0.5f);
            // The sheet gathered round: its columns around, its rows pole to
            // pole, every point pushed in or out by ridged noise -- the lumps.
            const float THETA = (S * 0.94f + 0.03f) * 2.0f * PI;
            const float PHI   = std::acos(std::clamp(1.0f - 2.0f * T, -1.0f, 1.0f));
            const Vec3  DIR{std::sin(PHI) * std::cos(THETA), -std::cos(PHI), std::sin(PHI) * std::sin(THETA)};
            const float LUMP = 1.0f - std::fabs(crumpleNoise(DIR.x * 2.3f + SEED, DIR.y * 2.3f, DIR.z * 2.3f - SEED));
            const Vec3  BALL = DIR * (R * (0.72f + 0.45f * LUMP));
            const Vec3  L    = flat + (BALL - flat) * K;
            P[static_cast<size_t>(j * (NX + 1) + i)] = CENTER +
                Vec3{ROT.m[0] * L.x + ROT.m[4] * L.y + ROT.m[8] * L.z, ROT.m[1] * L.x + ROT.m[5] * L.y + ROT.m[9] * L.z,
                     ROT.m[2] * L.x + ROT.m[6] * L.y + ROT.m[10] * L.z};
        }

    // Facets by brightness: 16 steps, one draw each.
    constexpr int LEVELS = 16;
    std::array<std::vector<float>, LEVELS> byShade;
    const Vec3 EYE = m_camera.position;
    const auto UVOF = [&w](int i, int j) {
        return Vec2{w.u0 + (w.u1 - w.u0) * (static_cast<float>(i) / NX), w.v0 + (w.v1 - w.v0) * (static_cast<float>(j) / NY)};
    };
    const auto TRI = [&](int ia, int ja, int ib, int jb, int ic, int jc) {
        const Vec3& A = P[static_cast<size_t>(ja * (NX + 1) + ia)];
        const Vec3& B = P[static_cast<size_t>(jb * (NX + 1) + ib)];
        const Vec3& C = P[static_cast<size_t>(jc * (NX + 1) + ic)];
        const Vec3  N = cross(B - A, C - A);
        const float LEN = std::sqrt(dot(N, N));
        if (LEN < 1e-9f)
            return;
        const Vec3  MID = (A + B + C) * (1.0f / 3.0f);
        const Vec3  TO  = normalize(EYE - MID);
        const float LIT = std::fabs(dot(N * (1.0f / LEN), TO));
        // Flat it reads as the window it is; crumpled, the facets turn away.
        const float SHADE = 1.0f - K * (0.62f * (1.0f - LIT));
        const int   LEVEL = std::clamp(static_cast<int>(std::lround(SHADE * (LEVELS - 1))), 0, LEVELS - 1);
        const Vec2  UA = UVOF(ia, ja), UB = UVOF(ib, jb), UC = UVOF(ic, jc);
        byShade[static_cast<size_t>(LEVEL)].insert(byShade[static_cast<size_t>(LEVEL)].end(),
            {A.x, A.y, A.z, UA.x, UA.y, B.x, B.y, B.z, UB.x, UB.y, C.x, C.y, C.z, UC.x, UC.y});
    };
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            TRI(i, j, i + 1, j, i + 1, j + 1);
            TRI(i, j, i + 1, j + 1, i, j + 1);
        }

    ensureCrumpleBuffers();

    glUseProgram(m_sceneProgram);
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, vp.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTexture, 0);
    glUniform1i(m_sceneTextured, 1);
    glUniform1f(m_sceneLodBias, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, w.texture);
    glBindVertexArray(m_crumpleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_crumpleVBO);
    for (int l = 0; l < LEVELS; ++l) {
        const auto& V = byShade[static_cast<size_t>(l)];
        if (V.empty())
            continue;
        const float SH = static_cast<float>(l) / (LEVELS - 1);
        glUniform4f(m_sceneColorUniform, SH, SH, SH, w.alpha);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(V.size() * sizeof(float)), V.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLint>(V.size() / 5));
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDepthMask(GL_FALSE);
}

void GLScene::drawBinHalo(const Mat4& vp) {
    if (m_binHaloI <= 0.0f || m_binHaloR <= 0.0f)
        return;
    // Concentric flat rings around the rim, fading outward, added to what
    // is behind (light, not paint): a bright line on the rim and a soft
    // falloff a hand wide. The draft's accent, --larch-ac #5fd3c4.
    constexpr int   SEG   = 64;
    constexpr int   RINGS = 6;
    constexpr float PI    = 3.14159265f;
    ensureCrumpleBuffers();
    glUseProgram(m_sceneProgram);
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, vp.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTextured, 0);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
    glBindVertexArray(m_crumpleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_crumpleVBO);
    std::vector<float> V;
    for (int r = 0; r < RINGS; ++r) {
        const float IN  = m_binHaloR - 0.012f + 0.016f * static_cast<float>(r);
        const float OUT = IN + 0.016f;
        V.clear();
        for (int i = 0; i <= SEG; ++i) {
            const float A = 2.0f * PI * static_cast<float>(i) / SEG;
            const float C = std::cos(A), S = std::sin(A);
            V.insert(V.end(), {m_binHaloAt.x + C * IN, m_binHaloAt.y, m_binHaloAt.z + S * IN, 0.f, 0.f,
                               m_binHaloAt.x + C * OUT, m_binHaloAt.y, m_binHaloAt.z + S * OUT, 0.f, 0.f});
        }
        const float FALL = r == 0 ? 1.0f : 0.55f * std::pow(0.62f, static_cast<float>(r - 1));
        glUniform4f(m_sceneColorUniform, 0x5f / 255.f, 0xd3 / 255.f, 0xc4 / 255.f, std::min(1.0f, 0.5f * m_binHaloI) * FALL);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(V.size() * sizeof(float)), V.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<GLint>(V.size() / 5));
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
}

void GLScene::drawOverlay(int width, int height) {
    // Textures of images no longer shown go first.
    std::erase_if(m_overlayTex, [&](const auto& T) {
        const bool LIVE = std::ranges::any_of(m_overlay, [&](const SOverlaySprite& S) { return S.image && S.image->serial == T.first; });
        if (!LIVE)
            glDeleteTextures(1, &T.second);
        return !LIVE;
    });
    if (m_overlay.empty() || width <= 0 || height <= 0)
        return;
    if (!m_overlayVAO) {
        glGenVertexArrays(1, &m_overlayVAO);
        glGenBuffers(1, &m_overlayVBO);
        glBindVertexArray(m_overlayVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_overlayVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
    }
    glUseProgram(m_sceneProgram);
    const Mat4 I = Mat4::identity();
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, I.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTexture, 0);
    glUniform1i(m_sceneTextured, 1);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(m_overlayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_overlayVBO);
    // The images are premultiplied.
    glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    for (const auto& S : m_overlay) {
        if (!S.image || S.image->w <= 0 || S.image->h <= 0 || S.alpha <= 0.002f)
            continue;
        auto& tex = m_overlayTex[S.image->serial];
        if (!tex) {
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, S.image->w, S.image->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, S.image->rgba.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        } else
            glBindTexture(GL_TEXTURE_2D, tex);
        // Whole pixels, so the painted 1 px lines stay sharp. The target's
        // y = -1 is its top.
        float atX = width / 2.0f, atY = height / 2.0f;
        if (S.anchor == EAnchor::Top)
            atY = 0.0f;
        else if (S.anchor == EAnchor::TopLeft)
            atX = atY = 0.0f;
        else if (S.anchor == EAnchor::Bottom)
            atY = static_cast<float>(height);
        else if (S.anchor == EAnchor::Cursor || S.anchor == EAnchor::World) {
            // A world point through the primary view; the scene's NDC has y
            // up, the target's y down.
            if (S.anchor == EAnchor::Cursor && !m_cursorOn)
                continue;
            const auto& M = m_lastVP.m;
            const Vec3& P = S.anchor == EAnchor::Cursor ? m_cursorAt : S.at;
            const float X = M[0] * P.x + M[4] * P.y + M[8] * P.z + M[12];
            const float Y = M[1] * P.x + M[5] * P.y + M[9] * P.z + M[13];
            const float W = M[3] * P.x + M[7] * P.y + M[11] * P.z + M[15];
            if (W <= 1e-4f)
                continue;
            atX = (X / W + 1.0f) * 0.5f * width, atY = (1.0f - Y / W) * 0.5f * height;
        }
        const float X0 = std::round(atX + S.dx - S.image->ax);
        const float Y0 = std::round(atY + S.dy - S.image->ay);
        const float L = X0 / width * 2.f - 1.f, R = (X0 + S.image->w) / width * 2.f - 1.f;
        const float T = Y0 / height * 2.f - 1.f, B = (Y0 + S.image->h) / height * 2.f - 1.f;
        const float V[] = {L, T, 0.f, 0.f, 0.f, R, T, 0.f, 1.f, 0.f, R, B, 0.f, 1.f, 1.f,
                           L, T, 0.f, 0.f, 0.f, R, B, 0.f, 1.f, 1.f, L, B, 0.f, 0.f, 1.f};
        glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_DYNAMIC_DRAW);
        glUniform4f(m_sceneColorUniform, S.alpha, S.alpha, S.alpha, S.alpha);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
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
    drawBinHalo(vp);
    drawPortalPools(vp);
    if (primary && !sight)
        drawBackdrop();

    // Windows last, back to front in the BSP's exact order: crossing quads
    // are split along each other, and translucency composites in order. A
    // featured window (F2, F4) comes after the rest and the dark over them --
    // for the player's eyes: the companion's eye sees the room as it stands.
    if (!m_featuredId || sight) {
        drawWindows(vp, windows);
        // A featured window gone (the menu closed): its dark and blur clear
        // over the whole room.
        if (!sight && m_featuredBlur > 0.0f)
            drawBlur(m_featuredBlur);
        if (!sight && m_featuredDim > 0.0f)
            drawDim(m_featuredDim);
    } else {
        std::vector<WindowRender> rest, featured;
        for (const auto& W : windows)
            (W.id == m_featuredId ? featured : rest).push_back(W);
        drawWindows(vp, rest);
        if (m_featuredBlur > 0.0f)
            drawBlur(m_featuredBlur);
        if (m_featuredDim > 0.0f)
            drawDim(m_featuredDim);
        m_windowsOnTop = true;
        drawWindows(vp, featured);
        m_windowsOnTop = false;
    }

    if (!sight)
        drawFrameMarks(vp);

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
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glBindVertexArray(m_portalVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_portalVBO);
    glActiveTexture(GL_TEXTURE0);

    // Twice: first only what is clearly there, writing depth, so it hides
    // what stands behind; then its soft glow, which writes none and lands
    // only where the first pass drew nothing (the same depth fails LESS).
    for (int pass = 0; pass < 2; ++pass)
    for (const auto& P : m_portals) {
        if (!P.tex || P.aspect <= 0.0f)
            continue;
        glUniform1f(m_sceneAlphaCut, pass == 0 ? 0.35f : 0.004f);
        glDepthMask(pass == 0 ? GL_TRUE : GL_FALSE);
        // At a portal, the others step back into the dark.
        const float LIGHT = m_portalFocus.empty() || m_portalFocus == P.spec.name ? 1.0f : 0.45f;
        glUniform4f(m_sceneColorUniform, LIGHT, LIGHT, LIGHT, 1.f);
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
    glDepthMask(GL_TRUE);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLScene::drawPortalDive(int width, int height) {
    const auto& D = m_dive;
    if (D.alpha <= 0.002f || width <= 0 || height <= 0)
        return;
    const auto P = std::ranges::find_if(m_portals, [&](const SPortalGL& p) { return p.spec.name == D.name; });
    if (P == m_portals.end() || !P->tex)
        return;
    if (!m_overlayVAO) {
        glGenVertexArrays(1, &m_overlayVAO);
        glGenBuffers(1, &m_overlayVBO);
        glBindVertexArray(m_overlayVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_overlayVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
    }
    glUseProgram(m_sceneProgram);
    const Mat4 I = Mat4::identity();
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, I.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTexture, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(m_overlayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_overlayVBO);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    const auto quad = [&](float x0, float y0, float x1, float y1, float u0, float v0, float u1, float v1) {
        const float V[] = {x0, y0, 0.f, u0, v0, x1, y0, 0.f, u1, v0, x1, y1, 0.f, u1, v1,
                           x0, y0, 0.f, u0, v0, x1, y1, 0.f, u1, v1, x0, y1, 0.f, u0, v1};
        glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_DYNAMIC_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    };
    // The rim's glow under the cover, then the cover, then its 4 px line.
    const float PX = 2.0f / width, PY = 2.0f / height;
    glUniform1i(m_sceneTextured, 0);
    if (D.rimAlpha > 0.0f)
        for (int r = 6; r >= 1; --r) {
            const float G = static_cast<float>(r) * 28.0f / 6.0f;
            glUniform4f(m_sceneColorUniform, D.rim.x, D.rim.y, D.rim.z, D.rimAlpha * D.alpha * 0.10f);
            quad(D.x0 - G * PX, D.y0 - G * PY, D.x1 + G * PX, D.y1 + G * PY, 0.f, 0.f, 1.f, 1.f);
        }
    glUniform1i(m_sceneTextured, 1);
    glUniform1f(m_sceneLodBias, D.blur);
    glUniform4f(m_sceneColorUniform, D.bright, D.bright, D.bright, D.alpha);
    glBindTexture(GL_TEXTURE_2D, P->tex);
    quad(D.x0, D.y0, D.x1, D.y1, D.u0, D.v0, D.u1, D.v1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUniform1f(m_sceneLodBias, 0.0f);
    if (D.rimAlpha > 0.0f) {
        glUniform1i(m_sceneTextured, 0);
        glUniform4f(m_sceneColorUniform, D.rim.x, D.rim.y, D.rim.z, D.rimAlpha * D.alpha);
        const float LX = 4.0f * PX, LY = 4.0f * PY;
        quad(D.x0 - LX, D.y0 - LY, D.x1 + LX, D.y0, 0.f, 0.f, 1.f, 1.f);
        quad(D.x0 - LX, D.y1, D.x1 + LX, D.y1 + LY, 0.f, 0.f, 1.f, 1.f);
        quad(D.x0 - LX, D.y0, D.x0, D.y1, 0.f, 0.f, 1.f, 1.f);
        quad(D.x1, D.y0, D.x1 + LX, D.y1, 0.f, 0.f, 1.f, 1.f);
    }
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLScene::drawPortalPools(const Mat4& vp) {
    // Each lit portal throws a pool of its colour on the floor, so the doors
    // tell apart before they can be read: an ellipse half again as wide as
    // the face, a third as deep, its light falling off from the middle to
    // nothing at the edge, added to the floor. Rings of a mesh, as the bin's.
    if (std::ranges::none_of(m_portals, [](const SPortalGL& P) { return P.spec.lit; }))
        return;
    constexpr int   SEG = 48, RINGS = 24;
    constexpr float PI  = 3.14159265f;
    ensureCrumpleBuffers();
    glUseProgram(m_sceneProgram);
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, vp.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTextured, 0);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE, GL_ZERO, GL_ONE);
    glBindVertexArray(m_crumpleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_crumpleVBO);
    std::vector<float> V;
    for (const auto& P : m_portals) {
        if (!P.spec.lit)
            continue;
        const Vec3  FACE{std::sin(P.spec.yaw), 0.f, -std::cos(P.spec.yaw)};
        const Vec3  RIGHT = normalize(cross(FACE * -1.0f, Vec3{0.f, 1.f, 0.f}));
        const float AX = 0.74f * P.spec.width, AZ = 0.28f * P.spec.width;
        const Vec3  AT = P.spec.base + Vec3{0.f, 0.01f, 0.f}; // a hair over the floor
        const float STRENGTH = m_portalFocus == P.spec.name ? (m_portalFlood ? 1.0f : 0.7f) : m_portalFocus.empty() ? 0.45f : 0.2f;
        for (int r = 0; r < RINGS; ++r) {
            const float IN = static_cast<float>(r) / RINGS, OUT = static_cast<float>(r + 1) / RINGS;
            V.clear();
            for (int i = 0; i <= SEG; ++i) {
                const float A = 2.0f * PI * static_cast<float>(i) / SEG;
                const float C = std::cos(A), S = std::sin(A);
                const Vec3  PI_ = AT + RIGHT * (C * AX * IN) + FACE * (S * AZ * IN);
                const Vec3  PO  = AT + RIGHT * (C * AX * OUT) + FACE * (S * AZ * OUT);
                V.insert(V.end(), {PI_.x, PI_.y, PI_.z, 0.f, 0.f, PO.x, PO.y, PO.z, 0.f, 0.f});
            }
            glUniform4f(m_sceneColorUniform, P.spec.color.x, P.spec.color.y, P.spec.color.z,
                        STRENGTH * (1.0f - (IN + OUT) * 0.5f));
            glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(V.size() * sizeof(float)), V.data(), GL_DYNAMIC_DRAW);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<GLint>(V.size() / 5));
        }
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
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

void GLScene::drawBackdrop() {
    if (m_backdropAlpha <= 0.001f || !m_backdrop.texture)
        return;
    if (!m_hudVAO) {
        glGenVertexArrays(1, &m_hudVAO);
        glGenBuffers(1, &m_hudVBO);
        glBindVertexArray(m_hudVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_hudVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
    }
    const GLboolean DEPTH = glIsEnabled(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glUseProgram(m_sceneProgram);
    const Mat4 I = Mat4::identity();
    glUniformMatrix4fv(m_sceneMVP, 1, GL_FALSE, I.m.data());
    glUniform4f(m_sceneUVRect, 0.f, 0.f, 1.f, 1.f);
    glUniform1i(m_sceneTexture, 0);
    glUniform1i(m_sceneTextured, 1);
    glUniform4f(m_sceneColorUniform, 1.f, 1.f, 1.f, std::clamp(m_backdropAlpha, 0.0f, 1.0f));
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(m_hudVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_hudVBO);
    // The scene's y is up, the HUD's down: its top at +1 here.
    const auto& Q = m_backdrop;
    const float V[] = {Q.x0, -Q.y0, 0.f, Q.u0, Q.v0, Q.x1, -Q.y0, 0.f, Q.u1, Q.v0, Q.x1, -Q.y1, 0.f, Q.u1, Q.v1,
                       Q.x0, -Q.y0, 0.f, Q.u0, Q.v0, Q.x1, -Q.y1, 0.f, Q.u1, Q.v1, Q.x0, -Q.y1, 0.f, Q.u0, Q.v1};
    glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_DYNAMIC_DRAW);
    glBindTexture(GL_TEXTURE_2D, Q.texture);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    if (DEPTH)
        glEnable(GL_DEPTH_TEST);
}

void GLScene::drawBlur(float amount) {
    if (!m_sceneFBO || !m_sceneColor || m_sceneWidth <= 0 || m_sceneHeight <= 0)
        return;
    if (!m_blurDownProgram) {
        static constexpr const char* VS = R"GLSL(#version 300 es
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); vUV = aUV; }
)GLSL";
        // The dual filter's two halves (Bjørge, "Bandwidth-Efficient
        // Rendering", SIGGRAPH 2015): five taps going down, eight going up.
        static constexpr const char* DOWN = R"GLSL(#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
uniform vec2 uHalf;
out vec4 fragColor;
void main() {
    vec4 s = texture(uTex, vUV) * 4.0;
    s += texture(uTex, vUV - uHalf);
    s += texture(uTex, vUV + uHalf);
    s += texture(uTex, vUV + vec2(uHalf.x, -uHalf.y));
    s += texture(uTex, vUV - vec2(uHalf.x, -uHalf.y));
    fragColor = s / 8.0;
}
)GLSL";
        static constexpr const char* UP = R"GLSL(#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
uniform vec2 uHalf;
out vec4 fragColor;
void main() {
    vec4 s = texture(uTex, vUV + vec2(-uHalf.x * 2.0, 0.0));
    s += texture(uTex, vUV + vec2(-uHalf.x, uHalf.y)) * 2.0;
    s += texture(uTex, vUV + vec2(0.0, uHalf.y * 2.0));
    s += texture(uTex, vUV + vec2(uHalf.x, uHalf.y)) * 2.0;
    s += texture(uTex, vUV + vec2(uHalf.x * 2.0, 0.0));
    s += texture(uTex, vUV + vec2(uHalf.x, -uHalf.y)) * 2.0;
    s += texture(uTex, vUV + vec2(0.0, -uHalf.y * 2.0));
    s += texture(uTex, vUV + vec2(-uHalf.x, -uHalf.y)) * 2.0;
    fragColor = s / 12.0;
}
)GLSL";
        const GLuint V = compileShader(GL_VERTEX_SHADER, VS);
        const GLuint D = compileShader(GL_FRAGMENT_SHADER, DOWN);
        const GLuint U = compileShader(GL_FRAGMENT_SHADER, UP);
        if (V && D)
            m_blurDownProgram = linkProgram(V, D);
        if (V && U)
            m_blurUpProgram = linkProgram(V, U);
        for (const GLuint SH : {V, D, U})
            if (SH)
                glDeleteShader(SH);
        if (!m_blurDownProgram || !m_blurUpProgram)
            return;
    }
    // Half, quarter and eighth of the room's size.
    for (int i = 0; i < 3; ++i) {
        auto&     L = m_blurLevels[i];
        const int W = std::max(1, m_sceneWidth >> (i + 1)), H = std::max(1, m_sceneHeight >> (i + 1));
        if (L.fbo && L.w == W && L.h == H)
            continue;
        if (!L.fbo) {
            glGenFramebuffers(1, &L.fbo);
            glGenTextures(1, &L.tex);
        }
        L.w = W, L.h = H;
        glBindTexture(GL_TEXTURE_2D, L.tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, L.fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, L.tex, 0);
    }

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    const GLboolean DEPTH = glIsEnabled(GL_DEPTH_TEST), BLEND = glIsEnabled(GL_BLEND), CULL = glIsEnabled(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(m_fullscreenVAO);
    // The spread of each tap, in source pixels: three levels at 2.5 come
    // near the draft's 14 px blur on its 1280 px stage at 1920 px.
    constexpr float SPREAD = 2.5f;
    const auto pass = [&](GLuint program, GLuint fbo, int w, int h, GLuint tex, int tw, int th) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, w, h);
        glUseProgram(program);
        glUniform1i(glGetUniformLocation(program, "uTex"), 0);
        glUniform2f(glGetUniformLocation(program, "uHalf"), SPREAD * 0.5f / tw, SPREAD * 0.5f / th);
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    };
    pass(m_blurDownProgram, m_blurLevels[0].fbo, m_blurLevels[0].w, m_blurLevels[0].h, m_sceneColor, m_sceneWidth, m_sceneHeight);
    for (int i = 1; i < 3; ++i)
        pass(m_blurDownProgram, m_blurLevels[i].fbo, m_blurLevels[i].w, m_blurLevels[i].h, m_blurLevels[i - 1].tex,
             m_blurLevels[i - 1].w, m_blurLevels[i - 1].h);
    for (int i = 2; i > 0; --i)
        pass(m_blurUpProgram, m_blurLevels[i - 1].fbo, m_blurLevels[i - 1].w, m_blurLevels[i - 1].h, m_blurLevels[i].tex,
             m_blurLevels[i].w, m_blurLevels[i].h);
    // Back into the room, mixed over it by `amount`; its alpha stays.
    glEnable(GL_BLEND);
    glBlendColor(0.f, 0.f, 0.f, std::clamp(amount, 0.0f, 1.0f));
    glBlendFuncSeparate(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA, GL_ZERO, GL_ONE);
    pass(m_blurUpProgram, m_sceneFBO, m_sceneWidth, m_sceneHeight, m_blurLevels[0].tex, m_blurLevels[0].w, m_blurLevels[0].h);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    if (!BLEND)
        glDisable(GL_BLEND);
    if (DEPTH)
        glEnable(GL_DEPTH_TEST);
    if (CULL)
        glEnable(GL_CULL_FACE);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
}

void GLScene::releaseBlur() {
    for (auto& L : m_blurLevels) {
        if (L.fbo)
            glDeleteFramebuffers(1, &L.fbo);
        if (L.tex)
            glDeleteTextures(1, &L.tex);
        L = {};
    }
    for (auto* P : {&m_blurDownProgram, &m_blurUpProgram, &m_frameProgram})
        if (*P) {
            glDeleteProgram(*P);
            *P = 0;
        }
    if (m_frameVAO)
        glDeleteVertexArrays(1, &m_frameVAO);
    if (m_frameVBO)
        glDeleteBuffers(1, &m_frameVBO);
    m_frameVAO = m_frameVBO = 0;
}

void GLScene::drawFrameMarks(const Mat4& vp) {
    if (m_frameMarks.empty() || m_sceneHeight <= 0)
        return;
    if (!m_frameProgram) {
        static constexpr const char* VS = R"GLSL(#version 300 es
precision highp float;
layout(location = 0) in vec2 aP;
uniform mat4 uMVP;
uniform vec3 uC, uR, uU;
out vec2 vP;
void main() {
    vP = aP;
    gl_Position = uMVP * vec4(uC + uR * aP.x + uU * aP.y, 1.0);
}
)GLSL";
        // Everything in the plane's local world units; uWpp turns screen
        // pixels into them (at the mark's centre). Layers composite back to front: fill, edge, brackets,
        // lines.
        static constexpr const char* FS = R"GLSL(#version 300 es
precision highp float;
in vec2 vP;
uniform vec2  uHalf;
uniform vec4  uContent; // left, top, right, bottom
uniform float uWpp;
uniform int   uFill;
uniform float uFillA;
uniform vec4  uColor; // the edge: rgb, alpha
uniform int   uDashed;
uniform float uBrackets;
uniform int   uHot;
uniform vec3  uHotColor;
uniform int   uLines;
uniform vec4  uLineP[2];
uniform vec4  uLineC[2];
out vec4 fragColor;
vec4 acc = vec4(0.0);
void over(vec3 c, float a) {
    a = clamp(a, 0.0, 1.0);
    acc.rgb = c * a + acc.rgb * (1.0 - a);
    acc.a   = a + acc.a * (1.0 - a);
}
float sdBox(vec2 p, vec2 c, vec2 h) {
    vec2 d = abs(p - c) - h;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}
// Coverage of a shape at signed distance d (world), antialiased over a pixel.
float cover(float d) {
    float aa = max(length(fwidth(vP)), 1e-6) * 0.7;
    return 1.0 - smoothstep(-aa, aa, d);
}
void main() {
    vec2 q = abs(vP);
    if (uFill != 0 && q.x <= uHalf.x && q.y <= uHalf.y) {
        bool inContent = vP.x >= uContent.x && vP.x <= uContent.z && vP.y <= uContent.y && vP.y >= uContent.w;
        if (!inContent) {
            if (uFill == 1) {
                // 135 degree stripes, 5 px of 12 on screen, over the window's
                // grey.
                vec2  w = vP / uWpp;
                float s = mod(w.x + w.y, 12.0 * 1.41421);
                over(mix(vec3(0.173, 0.180, 0.192), vec3(0.212, 0.220, 0.235), step(s, 5.0 * 1.41421)), uFillA);
            } else
                over(uColor.rgb, uFillA);
        }
    }
    if (uColor.a > 0.0) {
        float sd  = sdBox(vP, vec2(0.0), uHalf);
        float a   = cover(abs(sd) - 0.75 * uWpp);
        if (uDashed == 1) {
            float along = (q.x - uHalf.x > q.y - uHalf.y) ? vP.y : vP.x;
            a *= step(mod(along / uWpp, 14.0), 8.0);
        }
        over(uColor.rgb, a * uColor.a);
    }
    if (uBrackets > 0.0) {
        for (int i = 0; i < 4; ++i) {
            vec2  sg  = vec2(i == 1 || i == 3 ? 1.0 : -1.0, i < 2 ? 1.0 : -1.0);
            bool  HOT = i == uHot;
            float L   = (HOT ? 22.0 : 14.0) * uWpp, T = (HOT ? 3.0 : 2.0) * uWpp;
            vec2  C   = sg * (uHalf + 3.0 * uWpp);
            float d   = min(sdBox(vP, C - sg * vec2(L, T) * 0.5, vec2(L, T) * 0.5),
                            sdBox(vP, C - sg * vec2(T, L) * 0.5, vec2(T, L) * 0.5));
            over(HOT ? uHotColor : vec3(0.92, 0.93, 0.94), cover(d) * uBrackets * (HOT ? 1.0 : 0.75));
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (i >= uLines)
            break;
        vec2  A = uLineP[i].xy, B = uLineP[i].zw, AB = B - A;
        float h = clamp(dot(vP - A, AB) / max(dot(AB, AB), 1e-12), 0.0, 1.0);
        float d = length(vP - A - AB * h) - 0.65 * uWpp;
        float dash = step(mod(h * length(AB) / uWpp, 12.0), 7.0);
        over(uLineC[i].rgb, cover(d) * dash * uLineC[i].a);
    }
    if (acc.a <= 0.001)
        discard;
    fragColor = vec4(acc.rgb / acc.a, acc.a);
}
)GLSL";
        const GLuint V = compileShader(GL_VERTEX_SHADER, VS);
        const GLuint F = compileShader(GL_FRAGMENT_SHADER, FS);
        if (V && F)
            m_frameProgram = linkProgram(V, F);
        for (const GLuint SH : {V, F})
            if (SH)
                glDeleteShader(SH);
        if (!m_frameProgram)
            return;
        glGenVertexArrays(1, &m_frameVAO);
        glGenBuffers(1, &m_frameVBO);
        glBindVertexArray(m_frameVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_frameVBO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    const GLuint P = m_frameProgram;
    glUseProgram(P);
    glUniformMatrix4fv(glGetUniformLocation(P, "uMVP"), 1, GL_FALSE, vp.m.data());
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    // In front of the window it marks, which lies in the same plane.
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-1.f, -8.f);
    glBindVertexArray(m_frameVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_frameVBO);
    const float TAN = std::tan(kFovDeg * 3.14159265f / 360.0f);
    for (const auto& M : m_frameMarks) {
        const Vec3  D    = M.center - m_camera.position;
        const float DIST = std::max(0.05f, std::sqrt(D.x * D.x + D.y * D.y + D.z * D.z));
        const float WPP  = 2.0f * DIST * TAN / static_cast<float>(m_sceneHeight);
        // The quad: the frame, its brackets and its lines, a margin round.
        float x0 = -M.halfW, x1 = M.halfW, y0 = -M.halfH, y1 = M.halfH;
        for (const auto& L : M.lines)
            x0 = std::min({x0, L.x0, L.x1}), x1 = std::max({x1, L.x0, L.x1}), y0 = std::min({y0, L.y0, L.y1}), y1 = std::max({y1, L.y0, L.y1});
        const float MG = 10.0f * WPP;
        x0 -= MG, x1 += MG, y0 -= MG, y1 += MG;
        const float V[] = {x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1};
        glBufferData(GL_ARRAY_BUFFER, sizeof(V), V, GL_DYNAMIC_DRAW);
        glUniform3f(glGetUniformLocation(P, "uC"), M.center.x, M.center.y, M.center.z);
        glUniform3f(glGetUniformLocation(P, "uR"), M.right.x, M.right.y, M.right.z);
        glUniform3f(glGetUniformLocation(P, "uU"), M.up.x, M.up.y, M.up.z);
        glUniform2f(glGetUniformLocation(P, "uHalf"), M.halfW, M.halfH);
        glUniform4f(glGetUniformLocation(P, "uContent"), M.contentL, M.contentT, M.contentR, M.contentB);
        glUniform1f(glGetUniformLocation(P, "uWpp"), WPP);
        glUniform1i(glGetUniformLocation(P, "uFill"), static_cast<int>(M.fill));
        glUniform1f(glGetUniformLocation(P, "uFillA"), M.fillAlpha);
        glUniform4f(glGetUniformLocation(P, "uColor"), M.color.x, M.color.y, M.color.z, M.edge);
        glUniform1i(glGetUniformLocation(P, "uDashed"), M.dashed ? 1 : 0);
        glUniform1f(glGetUniformLocation(P, "uBrackets"), M.brackets);
        glUniform1i(glGetUniformLocation(P, "uHot"), M.hot);
        glUniform3f(glGetUniformLocation(P, "uHotColor"), M.hotColor.x, M.hotColor.y, M.hotColor.z);
        const int N = static_cast<int>(std::min<size_t>(M.lines.size(), 2));
        float     LP[8] = {}, LC[8] = {};
        for (int i = 0; i < N; ++i) {
            const auto& L = M.lines[static_cast<size_t>(i)];
            LP[i * 4 + 0] = L.x0, LP[i * 4 + 1] = L.y0, LP[i * 4 + 2] = L.x1, LP[i * 4 + 3] = L.y1;
            LC[i * 4 + 0] = L.color.x, LC[i * 4 + 1] = L.color.y, LC[i * 4 + 2] = L.color.z, LC[i * 4 + 3] = L.alpha;
        }
        glUniform1i(glGetUniformLocation(P, "uLines"), N);
        glUniform4fv(glGetUniformLocation(P, "uLineP"), 2, LP);
        glUniform4fv(glGetUniformLocation(P, "uLineC"), 2, LC);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }
    glDisable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(0.f, 0.f);
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
        m_gun.draw(vp, width, height, m_gunState);
    if (primary)
        m_lastVP = vp;

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
    if (primary) {
        drawPortalDive(width, height);
        drawOverlay(width, height);
    }

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
    m_gun.shutdown();
    m_gunState = {};
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
    releaseBlur();
    if (m_shadowTex) {
        glDeleteTextures(1, &m_shadowTex);
        m_shadowTex = 0;
    }
    if (m_hudVAO) {
        glDeleteVertexArrays(1, &m_hudVAO);
        glDeleteBuffers(1, &m_hudVBO);
        m_hudVAO = m_hudVBO = 0;
    }
    for (const auto& [SERIAL, TEX] : m_overlayTex)
        glDeleteTextures(1, &TEX);
    m_overlayTex.clear();
    if (m_overlayVAO) {
        glDeleteVertexArrays(1, &m_overlayVAO);
        glDeleteBuffers(1, &m_overlayVBO);
        m_overlayVAO = m_overlayVBO = 0;
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
