#include "LaserGun.hpp"
#include "LaserPistol.hpp"
#include <GLES3/gl32.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace H3D {
namespace {
constexpr float PI = 3.14159265359f;
GLuint program(const char* vs, const char* fs, std::string& error) {
    GLuint shaders[2] = {glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER)};
    const char* sources[2] = {vs, fs};
    bool ok = true;
    for (int i = 0; i < 2; ++i) {
        glShaderSource(shaders[i], 1, &sources[i], nullptr);
        glCompileShader(shaders[i]);
        GLint compiled = 0; glGetShaderiv(shaders[i], GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            char log[2048]{}; glGetShaderInfoLog(shaders[i], sizeof log, nullptr, log);
            error = std::string("Laser shader: ") + log; ok = false;
        }
    }
    GLuint result = 0;
    if (ok) {
        result = glCreateProgram();
        for (auto s : shaders) glAttachShader(result, s);
        glLinkProgram(result);
        GLint linked = 0; glGetProgramiv(result, GL_LINK_STATUS, &linked);
        if (!linked) {
            char log[2048]{}; glGetProgramInfoLog(result, sizeof log, nullptr, log);
            error = std::string("Laser link: ") + log; glDeleteProgram(result); result = 0;
        }
    }
    for (auto s : shaders) glDeleteShader(s);
    if (!error.empty()) std::fprintf(stderr, "hypr3d: %s\n", error.c_str());
    return result;
}
Vec3 point(const Mat4& m, const Vec3& p) {
    return {m.m[0]*p.x+m.m[4]*p.y+m.m[8]*p.z+m.m[12],
            m.m[1]*p.x+m.m[5]*p.y+m.m[9]*p.z+m.m[13],
            m.m[2]*p.x+m.m[6]*p.y+m.m[10]*p.z+m.m[14]};
}
}

bool CLaserGun::initialize() {
    if (m_fxProgram && m_paperProgram) return true;
    if (!m_error.empty()) return false;
    static const char* vs = R"GLSL(#version 300 es
layout(location=0) in vec3 aPosition;
layout(location=1) in vec2 aUV;
uniform mat4 uMVP;
out vec2 vUV;
void main(){ vUV=aUV; gl_Position=uMVP*vec4(aPosition,1.0); }
)GLSL";
    static const char* fs = R"GLSL(#version 300 es
precision highp float;
in vec2 vUV;
uniform vec4 uColor;
uniform int uGlow;
out vec4 fragColor;
void main(){
 float d=uGlow!=0 ? length(vUV*2.0-1.0) : abs(vUV.y*2.0-1.0);
 float a=pow(max(0.0,1.0-d),2.0);
 fragColor=vec4(uColor.rgb,uColor.a*a);
}
)GLSL";
    static const char* pvs = R"GLSL(#version 300 es
precision highp float;
layout(location=0) in vec3 aPosition;
layout(location=1) in vec2 aUV;
uniform mat4 uMVP;
uniform vec2 uSize,uOrigin;
uniform float uProgress;
out vec2 vUV;
void main(){
 vUV=aUV;
 float aspect=uSize.x/uSize.y;
 float reach=length(vec2(max(uOrigin.x,1.0-uOrigin.x)*aspect,max(uOrigin.y,1.0-uOrigin.y)));
 float d=length((aUV-uOrigin)*vec2(aspect,1.0))-uProgress*(reach+0.09);
 vec3 p=aPosition;
 // Heat lifts and ripples the surviving edge of the sheet.
 p.z=exp(-d*d/0.006)*sin(aUV.x*25.0+uProgress*14.0)*0.06*min(uSize.x,uSize.y)*uProgress;
 gl_Position=uMVP*vec4(p,1.0);
}
)GLSL";
    static const char* pfs = R"GLSL(#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTexture;
uniform vec4 uUVRect;
uniform vec2 uSize,uOrigin;
uniform float uProgress;
out vec4 fragColor;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);
 return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}
void main(){
 vec4 c=texture(uTexture,uUVRect.xy+vUV*uUVRect.zw);
 if(c.a<0.01 || uProgress>=1.0) discard;
 float aspect=uSize.x/uSize.y;
 float reach=length(vec2(max(uOrigin.x,1.0-uOrigin.x)*aspect,max(uOrigin.y,1.0-uOrigin.y)));
 float grain=noise(vUV*18.0)*0.045+noise(vUV*63.0)*0.025;
 float d=length((vUV-uOrigin)*vec2(aspect,1.0))+grain-uProgress*(reach+0.09);
 if(d<0.0) discard;
 float charred=1.0-smoothstep(0.01,0.065,d);
 c.rgb=mix(c.rgb,vec3(0.025,0.018,0.012),charred);
 float hot=1.0-smoothstep(0.006,0.025,d);
 c.rgb=mix(c.rgb,mix(vec3(1.0,0.18,0.025),vec3(1.0,0.9,0.4),hot),hot);
 fragColor=c;
}
)GLSL";
    m_fxProgram=program(vs,fs,m_error);
    m_paperProgram=program(pvs,pfs,m_error);
    if (!m_fxProgram || !m_paperProgram) return false;
    glGenVertexArrays(1,&m_vao); glGenBuffers(1,&m_vbo);
    glBindVertexArray(m_vao); glBindBuffer(GL_ARRAY_BUFFER,m_vbo);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,5*sizeof(float),nullptr);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(3*sizeof(float)));
    std::vector<float> grid;
    constexpr int X=32,Y=24;
    const auto add=[&](int x,int y){float s=float(x)/X,t=float(y)/Y;grid.insert(grid.end(),{s-0.5f,t-0.5f,0,s,t});};
    for(int y=0;y<Y;++y) for(int x=0;x<X;++x){add(x,y);add(x+1,y);add(x+1,y+1);add(x,y);add(x+1,y+1);add(x,y+1);}
    m_gridCount=grid.size()/5;
    glGenVertexArrays(1,&m_gridVAO);glGenBuffers(1,&m_gridVBO);
    glBindVertexArray(m_gridVAO);glBindBuffer(GL_ARRAY_BUFFER,m_gridVBO);
    glBufferData(GL_ARRAY_BUFFER,grid.size()*sizeof(float),grid.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,5*sizeof(float),nullptr);
    glEnableVertexAttribArray(1);glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(3*sizeof(float)));
    glBindVertexArray(0);
    return true;
}

void CLaserGun::vertices(const Mat4& vp,const float* data,size_t count,const Vec3& color,float alpha,bool isGlow) {
    glUseProgram(m_fxProgram);
    glUniformMatrix4fv(glGetUniformLocation(m_fxProgram,"uMVP"),1,GL_FALSE,vp.m.data());
    glUniform4f(glGetUniformLocation(m_fxProgram,"uColor"),color.x,color.y,color.z,alpha);
    glUniform1i(glGetUniformLocation(m_fxProgram,"uGlow"),isGlow);
    glBindVertexArray(m_vao);glBindBuffer(GL_ARRAY_BUFFER,m_vbo);
    glBufferData(GL_ARRAY_BUFFER,count*5*sizeof(float),data,GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES,0,count);glBindVertexArray(0);
}

void CLaserGun::glow(const Mat4& vp,const Vec3& p,float size,const Vec3& c,float a) {
    const float h=size*0.5f;
    const float v[]={p.x-h,p.y-h,p.z,0,0, p.x+h,p.y-h,p.z,1,0, p.x+h,p.y+h,p.z,1,1,
                     p.x-h,p.y-h,p.z,0,0, p.x+h,p.y+h,p.z,1,1, p.x-h,p.y+h,p.z,0,1};
    vertices(vp,v,6,c,a,true);
}

void CLaserGun::ribbon(const Mat4& vp,const Vec3& p,const Vec3& q,float width,const Vec3& c,float a) {
    Vec3 n=normalize(Vec3{-(q.y-p.y),q.x-p.x,0})*(width*0.5f);
    const Vec3 v[6]={p-n,p+n,q+n,p-n,q+n,q-n};
    float data[30];const float uv[12]={0,0,0,1,1,1,0,0,1,1,1,0};
    for(int i=0;i<6;++i){data[i*5]=v[i].x;data[i*5+1]=v[i].y;data[i*5+2]=v[i].z;data[i*5+3]=uv[i*2];data[i*5+4]=uv[i*2+1];}
    vertices(vp,data,6,c,a,false);
}

void CLaserGun::draw(const Mat4& worldVP,int width,int height,const State& s) {
    if(s.raised<=0.f || height<=0 || !initialize()) return;
    if(!m_started){
        m_model.setCenter(CMapModel::ECenter::Origin,{});
        m_model.load("<built-in laser pistol>",{},{},{1,1,1},kLaserPistol);
        m_started=true;
    }
    m_model.poll();
    if(m_model.failed()){m_error="Built-in laser pistol could not be decoded";return;}
    const float age=s.shotAge;
    const float recoil=age>=0.f && age<0.24f ? std::sin(std::min(age/0.045f,1.f)*PI*0.5f)*std::exp(-age*16.f) : 0.f;
    const float lift=1.f-s.raised;
    const float aspect=float(width)/height;
    const float fit=std::min(1.f,aspect/(4.f/3.f)); // portrait/square desktop outputs
    Vec3 pos{0.24f*fit,-0.14f-(1.f-fit)*0.08f-lift*0.32f,-0.48f+recoil*0.04f};
    Vec3 rot{8.f+recoil*14.f,18.f,-7.f};
    const float scale=0.32f*fit;
    const Mat4 model=Mat4::translation(pos)*Mat4::rotationY(rot.y*PI/180.f)*Mat4::rotationX(rot.x*PI/180.f)*Mat4::rotationZ(rot.z*PI/180.f)*Mat4::scale({scale,scale,scale});
    const Mat4 vp=Mat4::perspective(60.f*PI/180.f,float(width)/height,0.02f,20.f);
    // A separate depth pass keeps the tool out of room walls and windows.
    glClear(GL_DEPTH_BUFFER_BIT);glEnable(GL_DEPTH_TEST);glDepthMask(GL_TRUE);
    m_model.setTransform(pos,rot,{scale,scale,scale});m_model.draw(vp,{});
    const Vec3 muzzle=point(model,{0.f,0.055f,-0.37f});
    glEnable(GL_BLEND);glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE,GL_ZERO,GL_ONE);glDisable(GL_DEPTH_TEST);glDepthMask(GL_FALSE);
    const Vec3 energy=s.charged ? Vec3{1.f,0.3f,0.06f} : Vec3{0.1f,0.8f,1.f};
    if(s.charge>0.f){
        glow(vp,muzzle,(0.055f+0.09f*s.charge)*fit,energy,0.8f);
        glow(vp,muzzle,(0.025f+0.035f*s.charge)*fit,{1,1,1},s.charge);
    }
    const float shotLength=s.charged ? 0.32f : 0.18f;
    if(age>=0.f && age<shotLength){
        const auto& m=worldVP.m;const Vec3 p=s.target;
        const float w=m[3]*p.x+m[7]*p.y+m[11]*p.z+m[15];
        if(w>0.01f){
            const float x=(m[0]*p.x+m[4]*p.y+m[8]*p.z+m[12])/w;
            const float y=(m[1]*p.x+m[5]*p.y+m[9]*p.z+m[13])/w;
            const float depth=3.f,t=std::tan(30.f*PI/180.f);
            const Vec3 end{x*depth*t*float(width)/height,y*depth*t,-depth};
            const float fade=1.f-age/shotLength;
            ribbon(vp,muzzle,end,(s.charged?0.075f:0.028f)*fit,energy,fade);
            ribbon(vp,muzzle,end,(s.charged?0.023f:0.009f)*fit,{1,1,1},fade);
            glow(vp,muzzle,0.18f*fit,energy,fade);
            glow(vp,end,0.45f*fit,energy,fade);
        }
    }
    glDepthMask(GL_TRUE);glEnable(GL_DEPTH_TEST);
}

void CLaserGun::drawPaper(const Mat4& vp,const Paper& p) {
    if(!p.texture || p.progress>=1.f || !initialize()) return;
    const Mat4 pose=Mat4::translation(p.center)*Mat4::rotationY(p.yaw)*Mat4::rotationX(p.pitch)*Mat4::rotationZ(p.roll);
    const Mat4 mvp=vp*pose*Mat4::scale({p.width,p.height,1.f});
    glUseProgram(m_paperProgram);
    glUniformMatrix4fv(glGetUniformLocation(m_paperProgram,"uMVP"),1,GL_FALSE,mvp.m.data());
    glUniform2f(glGetUniformLocation(m_paperProgram,"uSize"),p.width,p.height);
    glUniform2f(glGetUniformLocation(m_paperProgram,"uOrigin"),p.origin.x,p.origin.y);
    glUniform1f(glGetUniformLocation(m_paperProgram,"uProgress"),p.progress);
    glUniform4f(glGetUniformLocation(m_paperProgram,"uUVRect"),p.u0,p.v0,p.u1-p.u0,p.v1-p.v0);
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,p.texture);
    glUniform1i(glGetUniformLocation(m_paperProgram,"uTexture"),0);
    glEnable(GL_DEPTH_TEST);glDepthMask(GL_TRUE);glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ZERO,GL_ONE);
    glBindVertexArray(m_gridVAO);glDrawArrays(GL_TRIANGLES,0,m_gridCount);glBindVertexArray(0);
    // Embers detach from the advancing front and rise like light paper ash.
    const float aspect=p.width/std::max(p.height,0.01f);
    const float reach=std::hypot(std::max(p.origin.x,1.f-p.origin.x)*aspect,std::max(p.origin.y,1.f-p.origin.y));
    glDepthMask(GL_FALSE);
    for(int i=0;i<32;++i){
        const float birth=0.02f+i*0.025f,life=p.progress-birth;
        if(life<0.f || life>0.3f) continue;
        const float angle=i*2.39996f,r=birth*(reach+0.09f);
        const float u=p.origin.x+std::cos(angle)*r/aspect,v=p.origin.y+std::sin(angle)*r;
        if(u<0 || u>1 || v<0 || v>1) continue;
        const Vec3 at=point(pose,{(u-.5f)*p.width+std::sin(i*7.f)*life*.2f,(v-.5f)*p.height+life*.7f,life*.15f});
        // World-aligned flecks; their small size and drift keep them light.
        glow(vp,at,0.014f+(i%3)*0.006f,life<0.12f?Vec3{1,.35f,.08f}:Vec3{.45f,.38f,.3f},(1.f-life/.3f)*.8f);
    }
    glDepthMask(GL_TRUE);
}

void CLaserGun::shutdown(){
    m_model.destroy();m_started=false;m_error.clear();
    if(m_fxProgram)glDeleteProgram(m_fxProgram);if(m_paperProgram)glDeleteProgram(m_paperProgram);
    if(m_vao)glDeleteVertexArrays(1,&m_vao);if(m_vbo)glDeleteBuffers(1,&m_vbo);
    if(m_gridVAO)glDeleteVertexArrays(1,&m_gridVAO);if(m_gridVBO)glDeleteBuffers(1,&m_gridVBO);
    m_fxProgram=m_paperProgram=m_vao=m_vbo=m_gridVAO=m_gridVBO=0;
}

} // namespace H3D
