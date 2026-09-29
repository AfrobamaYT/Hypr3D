#include "GLScene.hpp"

#include <GLES3/gl32.h>

#include <hyprgraphics/image/Image.hpp>

#include <algorithm>
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
uniform int uFrost;
uniform sampler2D uBlurTex;
uniform vec2 uScreen;

out vec4 fragColor;

void main() {
    if (uTextured != 0) {
        vec4 win = texture(uTexture, vUV) * uColor;

        // Frosted overlay: the blurred FINAL scene at this screen area,
        // alpha'd by the client content. The blur lands ON the window
        // itself (decoration:blur), not just the backdrop behind it.
        if (uFrost != 0) {
            vec3 bg = texture(uBlurTex, gl_FragCoord.xy / uScreen).rgb;
            fragColor = vec4(bg, win.a);
        } else {
            fragColor = win;
        }
    } else {
        fragColor = uColor;
    }
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

precision mediump float;

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

    static constexpr const char* blurVertexShader = R"GLSL(
#version 300 es

layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aUV;

out vec2 vUV;

void main() {
    vUV = aUV;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
)GLSL";

    // Separable 5-tap gaussian; the final pass applies the vibrancy boost.
    static constexpr const char* blurFragmentShader = R"GLSL(
#version 300 es

precision mediump float;

in vec2 vUV;

uniform sampler2D uTex;
uniform vec2 uTexel;
uniform vec2 uDir;
uniform float uRadius;
uniform float uVibrancy;
uniform int uFinal;

out vec4 fragColor;

void main() {
    vec4 c = texture(uTex, vUV) * 0.2270270270;

    vec2 o1 = uDir * uTexel * 1.3846153846 * uRadius;
    vec2 o2 = uDir * uTexel * 3.2307692308 * uRadius;

    c += texture(uTex, vUV + o1) * 0.3162162162;
    c += texture(uTex, vUV - o1) * 0.3162162162;
    c += texture(uTex, vUV + o2) * 0.0702702703;
    c += texture(uTex, vUV - o2) * 0.0702702703;

    if (uFinal != 0) {
        float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
        c.rgb = mix(vec3(luma), c.rgb, 1.0 + uVibrancy);
    }

    fragColor = c;
}
)GLSL";

    const GLuint BLUR_VS =
        compileShader(
            GL_VERTEX_SHADER,
            blurVertexShader
        );

    if (!BLUR_VS)
        return false;

    const GLuint BLUR_FS =
        compileShader(
            GL_FRAGMENT_SHADER,
            blurFragmentShader
        );

    if (!BLUR_FS) {
        glDeleteShader(BLUR_VS);
        return false;
    }

    m_blurProgram =
        linkProgram(
            BLUR_VS,
            BLUR_FS
        );

    glDeleteShader(BLUR_VS);
    glDeleteShader(BLUR_FS);

    if (!m_blurProgram)
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
uniform float uLod;
uniform sampler2D uPanorama;

out vec4 fragColor;

void main() {
    // Reconstruct the view ray for this pixel from the camera basis, then
    // sample the equirectangular panorama. fract() keeps u strictly inside
    // [0, 1) so the wrap point never rides the texture's outer edge.
    vec3 dir = normalize(
        uFwd + uRight * (vNdc.x * uTanHalfX) + uUp * (vNdc.y * uTanHalfY));

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

    m_sceneFrost =
        glGetUniformLocation(
            m_sceneProgram,
            "uFrost"
        );

    m_sceneBlurTex =
        glGetUniformLocation(
            m_sceneProgram,
            "uBlurTex"
        );

    m_sceneScreen =
        glGetUniformLocation(
            m_sceneProgram,
            "uScreen"
        );

    m_blurUTex =
        glGetUniformLocation(
            m_blurProgram,
            "uTex"
        );

    m_blurUTexel =
        glGetUniformLocation(
            m_blurProgram,
            "uTexel"
        );

    m_blurUDir =
        glGetUniformLocation(
            m_blurProgram,
            "uDir"
        );

    m_blurURadius =
        glGetUniformLocation(
            m_blurProgram,
            "uRadius"
        );

    m_blurUVibrancy =
        glGetUniformLocation(
            m_blurProgram,
            "uVibrancy"
        );

    m_blurUFinal =
        glGetUniformLocation(
            m_blurProgram,
            "uFinal"
        );

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
        m_sceneFrost >= 0 &&
        m_sceneBlurTex >= 0 &&
        m_sceneScreen >= 0 &&
        m_blurUTex >= 0 &&
        m_blurUTexel >= 0 &&
        m_blurUDir >= 0 &&
        m_blurURadius >= 0 &&
        m_blurUVibrancy >= 0 &&
        m_blurUFinal >= 0 &&
        m_blitTexture >= 0 &&
        m_blitAlpha >= 0 &&
        m_panoramaFwd >= 0 &&
        m_panoramaRight >= 0 &&
        m_panoramaUp >= 0 &&
        m_panoramaTanX >= 0 &&
        m_panoramaTanY >= 0 &&
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

    // Frosted overlay draw (window second pass only; the floor and crosshair
    // run with m_frost == false).
    glUniform1i(m_sceneFrost, m_frost ? 1 : 0);
    if (m_useBlur && m_blurFinalTex) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, m_blurFinalTex);
        glUniform1i(m_sceneBlurTex, 1);
        glUniform2f(m_sceneScreen,
            static_cast<float>(m_width), static_cast<float>(m_height));
        glActiveTexture(GL_TEXTURE0);
    }

    if (texture) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(m_sceneTexture, 0);
    }

    glBindVertexArray(vao);

    glDrawArrays(GL_TRIANGLES, 0, vertexCount);

    glBindVertexArray(0);

    if (m_useBlur && m_blurFinalTex) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
    }

    if (texture)
        glBindTexture(GL_TEXTURE_2D, 0);
}

void GLScene::drawFloor(
    const Mat4& vp
) {
    // Solid ground just under the grid. The floor is deliberately single-sided:
    // seeing this plane from below is a ceiling by definition, so back-face
    // culling makes the world coordinate system visually unambiguous.
    const Mat4 model =
        Mat4::translation({0.0f, FLOOR_Y - 0.02f, -4.0f}) *
        Mat4::scale({60.0f, 1.0f, 60.0f});

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

void GLScene::drawPanorama(float aspect) {
    if (!m_panoramaTex || !m_panoramaProgram)
        return;

    glUseProgram(m_panoramaProgram);

    const Vec3 FWD   = m_camera.forward();
    const Vec3 RIGHT = m_camera.right();
    const Vec3 UP    = cross(RIGHT, FWD);
    const float TANY = std::tan(65.0f * PI / 360.0f);

    // Analytic mip level: texels per screen pixel at the view centre. The
    // panorama is W texels around 2*pi radians; one screen pixel spans about
    // 2*tanX / screenWidth radians.
    float lod = 0.0f;

    if (m_panoramaW > 0 && m_sceneWidth > 0) {
        const float texelsPerPixel =
            static_cast<float>(m_panoramaW) / (2.0f * PI) *
            (2.0f * TANY * aspect) / static_cast<float>(m_sceneWidth);

        lod = std::clamp(std::log2(std::max(texelsPerPixel, 0.03125f)), 0.0f, 12.0f);
    }

    glUniform3f(m_panoramaFwd, FWD.x, FWD.y, FWD.z);
    glUniform3f(m_panoramaRight, RIGHT.x, RIGHT.y, RIGHT.z);
    glUniform3f(m_panoramaUp, UP.x, UP.y, UP.z);
    glUniform1f(m_panoramaTanX, TANY * aspect);
    glUniform1f(m_panoramaTanY, TANY);
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

bool GLScene::ensureBlurTargets(int width, int height) {
    const int W = std::max(1, width / 2);
    const int H = std::max(1, height / 2);

    if (m_blurFBOA && m_blurW == W && m_blurH == H)
        return true;

    destroyBlurTargets();

    glGenTextures(1, &m_blurTexA);
    glGenTextures(1, &m_blurTexB);
    glGenFramebuffers(1, &m_blurFBOA);
    glGenFramebuffers(1, &m_blurFBOB);

    for (unsigned tex : {m_blurTexA, m_blurTexB}) {
        glBindTexture(GL_TEXTURE_2D, tex);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }

    for (auto [fbo, tex] : {std::pair<unsigned, unsigned>{m_blurFBOA, m_blurTexA},
                            std::pair<unsigned, unsigned>{m_blurFBOB, m_blurTexB}}) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    m_blurW = W;
    m_blurH = H;

    return m_blurFBOA && m_blurFBOB;
}

void GLScene::destroyBlurTargets() {
    if (m_blurFBOA) {
        glDeleteFramebuffers(1, &m_blurFBOA);
        m_blurFBOA = 0;
    }
    if (m_blurFBOB) {
        glDeleteFramebuffers(1, &m_blurFBOB);
        m_blurFBOB = 0;
    }
    if (m_blurTexA) {
        glDeleteTextures(1, &m_blurTexA);
        m_blurTexA = 0;
    }
    if (m_blurTexB) {
        glDeleteTextures(1, &m_blurTexB);
        m_blurTexB = 0;
    }
    m_blurFinalTex = 0;
    m_blurW = 0;
    m_blurH = 0;
}

// Blurs the backdrop (already in m_sceneColor) into half-res ping-pong
// targets: one radius-0 copy, then `passes` separable H+V gaussian passes,
// the last one applying the vibrancy boost. The final texture is always
// m_blurTexA.
bool GLScene::renderBlur(int width, int height) {
    if (!ensureBlurTargets(width, height))
        return false;

    const float RADIUS = static_cast<float>(std::clamp(m_blurSize, 1, 16));

    glUseProgram(m_blurProgram);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    const auto pass = [&](unsigned srcTex, unsigned dstFBO, float dx, float dy,
                          float radius, bool finalPass) {
        glBindFramebuffer(GL_FRAMEBUFFER, dstFBO);
        glViewport(0, 0, m_blurW, m_blurH);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, srcTex);
        glUniform1i(m_blurUTex, 0);
        glUniform2f(m_blurUTexel, 1.0f / m_blurW, 1.0f / m_blurH);
        glUniform2f(m_blurUDir, dx, dy);
        glUniform1f(m_blurURadius, radius);
        glUniform1f(m_blurUVibrancy, m_blurVibrancy);
        glUniform1i(m_blurUFinal, finalPass ? 1 : 0);

        glBindVertexArray(m_fullscreenVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
    };

    // radius-0 copy of the sharp backdrop into the ping-pong chain
    pass(m_sceneColor, m_blurFBOA, 1, 0, 0, false);

    unsigned cur = m_blurTexA;

    for (int i = 0; i < m_blurPasses; ++i) {
        const bool LAST = i == m_blurPasses - 1;

        pass(cur, m_blurFBOB, 1, 0, RADIUS, false);
        pass(m_blurTexB, m_blurFBOA, 0, 1, RADIUS, LAST);
        cur = m_blurTexA;
    }

    m_blurFinalTex = cur;

    glBindTexture(GL_TEXTURE_2D, 0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE); // windows draw without depth writes, as before
    glViewport(0, 0, width, height);

    return true;
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
    const std::vector<WindowRender>& windows,
    bool frost
) {
    m_frost = frost;
    std::vector<const WindowRender*> visible;
    visible.reserve(windows.size());

    for (const auto& window : windows) {
        if (!window.texture)
            continue;

        if (window.alpha <= 0.0f)
            continue;

        if (window.width <= 0.0f || window.height <= 0.0f)
            continue;

        visible.emplace_back(&window);
    }

    if (visible.empty())
        return;

    // Alpha-blended windows must be submitted back to front, otherwise a
    // nearer window drawn first gets overpainted by the one behind it.
    const Vec3 eye = m_camera.position;

    std::sort(
        visible.begin(),
        visible.end(),
        [&eye](const WindowRender* a, const WindowRender* b) {
            const float dxA = a->x - eye.x, dyA = a->y - eye.y, dzA = a->z - eye.z;
            const float dxB = b->x - eye.x, dyB = b->y - eye.y, dzB = b->z - eye.z;

            const float distA = dxA * dxA + dyA * dyA + dzA * dzA;
            const float distB = dxB * dxB + dyB * dyB + dzB * dzB;

            return distA > distB;
        }
    );

    // Depth still occludes against the ground, but windows do not write depth
    // -- the ordering above is what decides their blending.
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        GL_ZERO, GL_ONE
    );

    for (const auto* window : visible) {
        const Mat4 model =
            Mat4::translation({window->x, window->y, window->z}) *
            Mat4::rotationY(window->yaw) *
            Mat4::rotationX(window->pitch) *
            Mat4::scale({window->width, window->height, 1.0f});

        const float uvRect[4] = {
            window->u0, window->v0, window->u1, window->v1
        };

        drawQuad(
            m_quadVAO,
            m_quadVertexCount,
            vp * model,
            window->texture,
            uvRect,
            1.0f, 1.0f, 1.0f, window->alpha
        );
    }

    glDepthMask(GL_TRUE);
}

void GLScene::drawCrosshair(int width, int height) {
    if (!m_crosshairVAO || !m_crosshairVBO || width <= 0 || height <= 0)
        return;

    const float pxX = 2.0f / static_cast<float>(width);
    const float pxY = 2.0f / static_cast<float>(height);

    constexpr float OUTER_PX   = 10.0f;
    constexpr float INNER_PX   = 3.5f;
    constexpr float THICK_PX   = 1.4f;
    constexpr float OUTLINE_PX = 1.0f;

    const auto addRect = [](std::vector<float>& v, float x0, float y0, float x1, float y1) {
        const float z = 0.0f;
        const float u = 0.0f;
        const float t = 0.0f;
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
    drawLayer(OUTER_PX + OUTLINE_PX, std::max(0.0f, INNER_PX - OUTLINE_PX),
        THICK_PX + 2.0f * OUTLINE_PX, 1.0f, 1.0f, 1.0f, 1.0f);
    drawLayer(OUTER_PX, INNER_PX, THICK_PX, 0.0f, 0.0f, 0.0f, 1.0f);
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

bool GLScene::render(
    unsigned int targetFBO,
    int width,
    int height,
    float alpha,
    float dt,
    const std::vector<WindowRender>& windows
) {
    if (width <= 0 || height <= 0)
        return false;

    if (!initialize())
        return false;

    if (!ensureSceneFramebuffer(width, height))
        return false;

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

    const Mat4 projection =
        Mat4::perspective(65.0f * PI / 180.0f, aspect, 0.05f, 200.0f);

    m_width  = width;
    m_height = height;

    const Mat4 vp = projection * m_camera.view();

    // Background panorama first, then the ground so windows behind it are
    // depth-rejected.
    refreshPanorama();
    drawPanorama(aspect);

    drawFloor(vp);

    glEnable(GL_BLEND);
    glBlendFuncSeparate(
        GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        GL_ZERO, GL_ONE
    );
    glDepthMask(GL_FALSE);

    drawGrid(vp);

    // Sharp pass first, then blur the FINAL scene (windows included) and draw
    // it back over the window areas: the blur lands ON the windows, not just
    // the backdrop behind them.
    drawWindows(vp, windows);

    m_useBlur = false;

    if (m_blurEnabled && renderBlur(width, height)) {
        m_useBlur = true;
        drawWindows(vp, windows, /*frost*/ true);
        m_useBlur = false;
    }

    glDepthMask(GL_TRUE);

    // Read back one pixel of the offscreen scene while it is still bound. A
    // floor point in the lower half of the screen, where ground and sky are
    // both opaque by construction: alpha below 255 here means the composite
    // cannot become fully opaque no matter what the fade is set to.
    if (m_probeRequested) {
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

void GLScene::setBlurConfig(bool enabled, int size, int passes, float vibrancy) {
    m_blurEnabled  = enabled;
    m_blurSize     = size;
    m_blurPasses   = passes;
    m_blurVibrancy = vibrancy;
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
    destroyGLObjects();
}

void GLScene::destroyGLObjects() {
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

    if (m_blurProgram) {
        glDeleteProgram(m_blurProgram);
        m_blurProgram = 0;
    }

    destroyBlurTargets();

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
