#pragma once

#include "Render/MapModel.hpp"
#include "World/Camera.hpp"
#include <string>

namespace H3D {

// First-person tool and the frozen paper left by its charged shot. No
// compositor objects here: main owns the target, process and snapshot.
class CLaserGun {
  public:
    struct State {
        float raised = 0.f, charge = 0.f, shotAge = -1.f;
        bool charged = false;
        Vec3 target{};
    };
    struct Paper {
        unsigned texture = 0;
        Vec3 center{};
        float width = 0.f, height = 0.f, yaw = 0.f, pitch = 0.f, roll = 0.f;
        float u0 = 0.f, v0 = 0.f, u1 = 1.f, v1 = 1.f;
        float progress = 0.f;
        Vec2 origin{0.5f, 0.5f}; // bottom-left, like the drawn quad
    };
    void draw(const Mat4& worldVP, int width, int height, const State& state);
    void drawPaper(const Mat4& vp, const Paper& paper);
    void shutdown();
    bool pending() const { return m_model.pending(); }
    bool loaded() const { return m_model.loaded(); }
    const std::string& error() const { return m_error; }

  private:
    bool initialize();
    void glow(const Mat4& vp, const Vec3& at, float size, const Vec3& color, float alpha);
    void ribbon(const Mat4& vp, const Vec3& from, const Vec3& to, float width,
                const Vec3& color, float alpha);
    void vertices(const Mat4& vp, const float* data, size_t count,
                  const Vec3& color, float alpha, bool glow);
    CMapModel m_model;
    bool m_started = false;
    std::string m_error;
    unsigned m_fxProgram = 0, m_paperProgram = 0;
    unsigned m_vao = 0, m_vbo = 0, m_gridVAO = 0, m_gridVBO = 0;
    int m_gridCount = 0;
};

} // namespace H3D
