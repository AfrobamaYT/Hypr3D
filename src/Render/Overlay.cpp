#include "Overlay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

#include <cairo/cairo.h>
#include <pango/pangocairo.h>

namespace Overlay {

    namespace {

        // The Larch design system's tokens (_ds_bundle.css), 0..1.
        struct SRGBA {
            double r, g, b, a = 1.0;
        };
        constexpr SRGBA hex(unsigned v, double a = 1.0) {
            return {((v >> 16) & 0xff) / 255.0, ((v >> 8) & 0xff) / 255.0, (v & 0xff) / 255.0, a};
        }
        constexpr SRGBA PANEL    = {20 / 255.0, 23 / 255.0, 26 / 255.0, 0.93};
        constexpr SRGBA LINE     = {1, 1, 1, 0.09};
        constexpr SRGBA LINE_KEY = {1, 1, 1, 0.12};
        constexpr SRGBA FILL_KEY = {1, 1, 1, 0.09};
        constexpr SRGBA HEADING  = hex(0xf1f4f6);
        constexpr SRGBA DOT      = hex(0x59636b);
        constexpr SRGBA CODE     = hex(0xe6ebef);
        constexpr SRGBA HINT     = hex(0xc3ccd3);
        constexpr SRGBA SUB      = hex(0xa3adb5);
        constexpr SRGBA TITLE    = hex(0xf4f6f8);
        constexpr SRGBA TEXT     = hex(0xeef2f5);
        constexpr SRGBA ERROR    = hex(0xff8a7a);
        constexpr SRGBA UI       = hex(0x5cb8e6);
        constexpr SRGBA LABEL    = hex(0x8a959e);
        constexpr SRGBA VALUE    = hex(0xd7dde2);
        constexpr const char* SANS = "Manrope";
        constexpr const char* MONO = "JetBrains Mono";

        void source(cairo_t* cr, const SRGBA& c, double alpha = 1.0) {
            cairo_set_source_rgba(cr, c.r, c.g, c.b, c.a * alpha);
        }

        void roundRect(cairo_t* cr, double x, double y, double w, double h, double r) {
            r = std::min({r, w / 2, h / 2});
            constexpr double PI = std::numbers::pi;
            cairo_new_sub_path(cr);
            cairo_arc(cr, x + w - r, y + r, r, -PI / 2, 0);
            cairo_arc(cr, x + w - r, y + h - r, r, 0, PI / 2);
            cairo_arc(cr, x + r, y + h - r, r, PI / 2, PI);
            cairo_arc(cr, x + r, y + r, r, PI, 3 * PI / 2);
            cairo_close_path(cr);
        }

        // One run of text: its layout, laid out on a scratch context and
        // re-targeted to the real one when drawn.
        struct SText {
            PangoLayout* layout = nullptr;
            double       w = 0, h = 0, baseline = 0;
            SRGBA        color{};
            SText() = default;
            SText(const SText&) = delete;
            SText(SText&& o) noexcept : layout(o.layout), w(o.w), h(o.h), baseline(o.baseline), color(o.color) {
                o.layout = nullptr;
            }
            SText& operator=(SText&& o) noexcept {
                std::swap(layout, o.layout);
                w = o.w, h = o.h, baseline = o.baseline, color = o.color;
                return *this;
            }
            ~SText() {
                if (layout)
                    g_object_unref(layout);
            }
        };

        cairo_t* scratch() {
            static cairo_surface_t* S  = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 1, 1);
            static cairo_t*         CR = cairo_create(S);
            return CR;
        }

        SText text(const std::string& str, int weight, double px, const SRGBA& color, double scale,
                   const char* family = SANS, double maxWidth = 0, double tracking = 0) {
            SText T;
            T.color  = color;
            T.layout = pango_cairo_create_layout(scratch());
            auto* OPTS = cairo_font_options_create();
            cairo_font_options_set_antialias(OPTS, CAIRO_ANTIALIAS_GRAY);
            cairo_font_options_set_hint_style(OPTS, CAIRO_HINT_STYLE_SLIGHT);
            cairo_font_options_set_hint_metrics(OPTS, CAIRO_HINT_METRICS_OFF);
            pango_cairo_context_set_font_options(pango_layout_get_context(T.layout), OPTS);
            cairo_font_options_destroy(OPTS);
            auto* FD = pango_font_description_new();
            pango_font_description_set_family(FD, family);
            pango_font_description_set_weight(FD, static_cast<PangoWeight>(weight));
            pango_font_description_set_absolute_size(FD, px * scale * PANGO_SCALE);
            pango_layout_set_font_description(T.layout, FD);
            pango_font_description_free(FD);
            pango_layout_set_text(T.layout, str.c_str(), -1);
            if (tracking > 0) {
                auto* ATTRS = pango_attr_list_new();
                pango_attr_list_insert(ATTRS, pango_attr_letter_spacing_new(static_cast<int>(tracking * scale * PANGO_SCALE)));
                pango_layout_set_attributes(T.layout, ATTRS);
                pango_attr_list_unref(ATTRS);
            }
            if (maxWidth > 0) {
                pango_layout_set_width(T.layout, static_cast<int>(maxWidth * scale * PANGO_SCALE));
                pango_layout_set_ellipsize(T.layout, PANGO_ELLIPSIZE_END);
            }
            pango_context_changed(pango_layout_get_context(T.layout));
            pango_layout_context_changed(T.layout);
            PangoRectangle ink, logical;
            pango_layout_get_pixel_extents(T.layout, &ink, &logical);
            T.w        = logical.width;
            T.h        = logical.height;
            T.baseline = pango_layout_get_baseline(T.layout) / static_cast<double>(PANGO_SCALE);
            return T;
        }

        void draw(cairo_t* cr, const SText& t, double x, double y, double alpha = 1.0) {
            cairo_move_to(cr, x, y);
            source(cr, t.color, alpha);
            pango_cairo_update_layout(cr, t.layout);
            pango_cairo_show_layout(cr, t.layout);
        }

        // Box blur of a float plane, rows then columns, `passes` times -- three
        // passes are close to a Gaussian.
        void blur(std::vector<float>& a, int w, int h, int r, int passes) {
            if (r <= 0)
                return;
            std::vector<float> line(static_cast<size_t>(std::max(w, h)));
            const float        N = 1.0f / static_cast<float>(2 * r + 1);
            for (int p = 0; p < passes; ++p) {
                for (int y = 0; y < h; ++y) {
                    float* row = &a[static_cast<size_t>(y) * w];
                    float  sum = 0;
                    for (int i = -r; i <= r; ++i)
                        sum += row[std::clamp(i, 0, w - 1)] * (i >= 0 && i < w ? 1.f : 0.f);
                    for (int x = 0; x < w; ++x) {
                        line[x] = sum * N;
                        const int ADD = x + r + 1, SUB = x - r;
                        sum += (ADD < w ? row[ADD] : 0.f) - (SUB >= 0 ? row[SUB] : 0.f);
                    }
                    std::copy_n(line.begin(), w, row);
                }
                for (int x = 0; x < w; ++x) {
                    float sum = 0;
                    for (int i = 0; i <= r && i < h; ++i)
                        sum += a[static_cast<size_t>(i) * w + x];
                    for (int y = 0; y < h; ++y) {
                        line[y] = sum * N;
                        const int ADD = y + r + 1, SUB = y - r;
                        sum += (ADD < h ? a[static_cast<size_t>(ADD) * w + x] : 0.f) -
                            (SUB >= 0 ? a[static_cast<size_t>(SUB) * w + x] : 0.f);
                    }
                    for (int y = 0; y < h; ++y)
                        a[static_cast<size_t>(y) * w + x] = line[y];
                }
            }
        }

        // A blur radius per pass for a Gaussian of `sigma`: a single 3-wide
        // box below about a pixel, three boxes above.
        void gaussian(std::vector<float>& a, int w, int h, double sigma) {
            if (sigma < 1.2) {
                blur(a, w, h, 1, 1);
                return;
            }
            const int R = std::max(1, static_cast<int>(std::lround((std::sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0)));
            blur(a, w, h, R, 3);
        }

        // Cairo's ARGB32 (native-endian words, premultiplied) to RGBA bytes,
        // a shadow under it: `shadow` is its alpha plane, already blurred, of
        // colour `sc`.
        SImage finish(cairo_surface_t* s, const std::vector<float>* shadow, const SRGBA& sc, double opacity) {
            cairo_surface_flush(s);
            static uint64_t serial = 0;
            SImage          I;
            I.serial = ++serial;
            I.w = cairo_image_surface_get_width(s);
            I.h = cairo_image_surface_get_height(s);
            const int STRIDE = cairo_image_surface_get_stride(s);
            const auto* DATA = cairo_image_surface_get_data(s);
            I.rgba.resize(static_cast<size_t>(I.w) * I.h * 4);
            for (int y = 0; y < I.h; ++y)
                for (int x = 0; x < I.w; ++x) {
                    const uint32_t P = *reinterpret_cast<const uint32_t*>(DATA + static_cast<size_t>(y) * STRIDE + x * 4);
                    float          a = ((P >> 24) & 0xff) / 255.f, r = ((P >> 16) & 0xff) / 255.f,
                          g = ((P >> 8) & 0xff) / 255.f, b = (P & 0xff) / 255.f;
                    if (shadow) {
                        const float SA = std::clamp((*shadow)[static_cast<size_t>(y) * I.w + x], 0.f, 1.f) * static_cast<float>(sc.a) * (1.f - a);
                        r += static_cast<float>(sc.r) * SA, g += static_cast<float>(sc.g) * SA, b += static_cast<float>(sc.b) * SA, a += SA;
                    }
                    const float O = static_cast<float>(opacity);
                    auto*       out = &I.rgba[(static_cast<size_t>(y) * I.w + x) * 4];
                    out[0] = static_cast<uint8_t>(std::lround(std::clamp(r * O, 0.f, 1.f) * 255.f));
                    out[1] = static_cast<uint8_t>(std::lround(std::clamp(g * O, 0.f, 1.f) * 255.f));
                    out[2] = static_cast<uint8_t>(std::lround(std::clamp(b * O, 0.f, 1.f) * 255.f));
                    out[3] = static_cast<uint8_t>(std::lround(std::clamp(a * O, 0.f, 1.f) * 255.f));
                }
            return I;
        }

        // The alpha of what is painted so far, as a float plane.
        std::vector<float> alphaOf(cairo_surface_t* s) {
            cairo_surface_flush(s);
            const int   W = cairo_image_surface_get_width(s), H = cairo_image_surface_get_height(s);
            const int   STRIDE = cairo_image_surface_get_stride(s);
            const auto* DATA = cairo_image_surface_get_data(s);
            std::vector<float> a(static_cast<size_t>(W) * H);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    a[static_cast<size_t>(y) * W + x] =
                        ((*reinterpret_cast<const uint32_t*>(DATA + static_cast<size_t>(y) * STRIDE + x * 4) >> 24) & 0xff) / 255.f;
            return a;
        }

        // A key cap: 28 high, at least 28 wide, 8 either side of its text;
        // the draft's fill, a 2 px lip at the bottom and a 1 px rim.
        struct SCapStyle {
            double h, minW, pad, radius;
            int    weight;
            double px;
        };
        constexpr SCapStyle CAP_SMALL = {28, 28, 8, 8, 800, 13};
        constexpr SCapStyle CAP       = {30, 32, 9, 9, 700, 14};

        double capWidth(const SText& t, const SCapStyle& st, double s) {
            return std::max(st.minW * s, t.w + 2 * st.pad * s);
        }

        void drawCap(cairo_t* cr, const SText& t, double x, double y, const SCapStyle& st, double s) {
            const double W = capWidth(t, st, s), H = st.h * s, R = st.radius * s;
            roundRect(cr, x, y, W, H, R);
            source(cr, FILL_KEY);
            cairo_fill(cr);
            roundRect(cr, x + 0.5 * s, y + 0.5 * s, W - s, H - s, R - 0.5 * s);
            source(cr, LINE_KEY);
            cairo_set_line_width(cr, s);
            cairo_stroke(cr);
            // inset 0 -2px 0 rgba(0,0,0,.35): what the cap shifted 2 px up
            // leaves uncovered at its bottom.
            cairo_save(cr);
            roundRect(cr, x, y, W, H, R);
            cairo_clip(cr);
            cairo_set_fill_rule(cr, CAIRO_FILL_RULE_EVEN_ODD);
            roundRect(cr, x, y, W, H, R);
            roundRect(cr, x, y - 2 * s, W, H, R);
            cairo_set_source_rgba(cr, 0, 0, 0, 0.35);
            cairo_fill(cr);
            cairo_restore(cr);
            draw(cr, t, x + (W - t.w) / 2, y + (H - t.h) / 2);
        }

        // A Larch panel: the fill and its 1 px inner line.
        void drawPanel(cairo_t* cr, double x, double y, double w, double h, double r, double s) {
            roundRect(cr, x, y, w, h, r);
            source(cr, PANEL);
            cairo_fill(cr);
            roundRect(cr, x + 0.5 * s, y + 0.5 * s, w - s, h - s, r - 0.5 * s);
            source(cr, LINE);
            cairo_set_line_width(cr, s);
            cairo_stroke(cr);
        }

        // shadow-float, 0 24px 70px rgba(0,0,0,.5): a blur of 70 px is a
        // Gaussian of sigma 35; it is cut at two sigmas, where it is 1 % of
        // its strength. Like CSS, it is not painted under the box itself.
        struct SMargins {
            double l, t, r, b;
        };
        SMargins floatMargins(double s) {
            return {70 * s, std::max(2.0, 46 * s), 70 * s, 94 * s};
        }

        std::vector<float> paintFloatShadow(int W, int H, double x, double y, double w, double h, double r, double s);

        // The shadow depends on the box alone, and F3's panel repaints 4 x a
        // second at one size: the last few are kept.
        std::vector<float> floatShadow(int W, int H, double x, double y, double w, double h, double r, double s) {
            struct SEntry {
                std::array<double, 8> key;
                std::vector<float>    a;
            };
            static std::vector<SEntry> cache;
            const std::array<double, 8> KEY = {double(W), double(H), x, y, w, h, r, s};
            for (const auto& E : cache)
                if (E.key == KEY)
                    return E.a;
            if (cache.size() >= 6)
                cache.erase(cache.begin());
            cache.push_back({KEY, paintFloatShadow(W, H, x, y, w, h, r, s)});
            return cache.back().a;
        }

        std::vector<float> paintFloatShadow(int W, int H, double x, double y, double w, double h, double r, double s) {
            auto* S  = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, W, H);
            auto* CR = cairo_create(S);
            roundRect(CR, x, y + 24 * s, w, h, r);
            cairo_set_source_rgba(CR, 0, 0, 0, 1);
            cairo_fill(CR);
            cairo_destroy(CR);
            auto A = alphaOf(S);
            gaussian(A, W, H, 35 * s);
            // Cut out the box itself.
            CR = cairo_create(S);
            cairo_set_operator(CR, CAIRO_OPERATOR_SOURCE);
            cairo_set_source_rgba(CR, 0, 0, 0, 0);
            cairo_paint(CR);
            cairo_set_operator(CR, CAIRO_OPERATOR_OVER);
            roundRect(CR, x, y, w, h, r);
            cairo_set_source_rgba(CR, 0, 0, 0, 1);
            cairo_fill(CR);
            cairo_destroy(CR);
            const auto BOX = alphaOf(S);
            cairo_surface_destroy(S);
            for (size_t i = 0; i < A.size(); ++i)
                A[i] *= 1.0f - BOX[i];
            return A;
        }
    }

    SImage paintMark(EMark mark, float scale, float charge) {
        if (mark == EMark::None)
            return {};
        const double s    = std::max(0.5f, scale);
        const int    SIZE = static_cast<int>(std::ceil(48 * s / 2)) * 2; // even: the centre on a pixel edge
        auto*        S    = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
        auto*        CR   = cairo_create(S);
        const double C    = SIZE / 2.0;
        cairo_translate(CR, C, C);
        cairo_scale(CR, s, s);
        constexpr double PI = std::numbers::pi;

        const auto dot = [&](const SRGBA& c, double r = 2) {
            cairo_arc(CR, 0, 0, r, 0, 2 * PI);
            source(CR, c);
            cairo_fill(CR);
        };
        const auto ring = [&](double outer, const SRGBA& c) {
            cairo_arc(CR, 0, 0, outer - 1, 0, 2 * PI);
            cairo_set_line_width(CR, 2);
            source(CR, c);
            cairo_stroke(CR);
        };
        const auto cross = [&](const SRGBA& c) {
            source(CR, c);
            cairo_rectangle(CR, -15, -1, 9, 2);
            cairo_rectangle(CR, 6, -1, 9, 2);
            cairo_rectangle(CR, -1, -15, 2, 9);
            cairo_rectangle(CR, -1, 6, 2, 9);
            cairo_fill(CR);
        };

        bool   shadow  = true;
        double opacity = 1.0;
        switch (mark) {
            case EMark::Window: {
                // Brackets in a window's shape: 24 px, corners 7 px, 2 px lines.
                source(CR, UI);
                for (const double SX : {-1.0, 1.0})
                    for (const double SY : {-1.0, 1.0}) {
                        const double X0 = SX < 0 ? -12 : 5, Y0 = SY < 0 ? -12 : 5;
                        cairo_rectangle(CR, X0, SY < 0 ? -12 : 10, 7, 2);
                        cairo_rectangle(CR, SX < 0 ? -12 : 10, Y0, 2, 7);
                    }
                cairo_fill(CR);
                dot(UI);
                break;
            }
            case EMark::Carry:
                ring(9, UI);
                dot(UI);
                break;
            case EMark::Fixed:
                ring(9, {TITLE.r, TITLE.g, TITLE.b, 0.7});
                dot(TITLE);
                break;
            case EMark::Portal:
                cairo_rotate(CR, PI / 4);
                roundRect(CR, -7, -7, 14, 14, 1);
                cairo_set_line_width(CR, 2);
                source(CR, UI);
                cairo_stroke(CR);
                cairo_rectangle(CR, -2, -2, 4, 4);
                cairo_fill(CR);
                break;
            case EMark::Nothing:
                // A 4 px dot at 75 % with a 1 px dark ring around it.
                shadow = false;
                cairo_arc(CR, 0, 0, 2.5, 0, 2 * PI);
                cairo_set_line_width(CR, 1);
                cairo_set_source_rgba(CR, 8 / 255.0, 9 / 255.0, 10 / 255.0, 0.6);
                cairo_stroke(CR);
                dot({TITLE.r, TITLE.g, TITLE.b, 0.75});
                break;
            case EMark::Bin:
                ring(11, ERROR);
                dot(ERROR);
                break;
            case EMark::Gun:
                cross(ERROR);
                dot(ERROR);
                break;
            case EMark::GunIdle:
                cross(TITLE);
                opacity = 0.55;
                break;
            case EMark::GunHold: {
                cairo_set_line_width(CR, 3);
                cairo_arc(CR, 0, 0, 17, 0, 2 * PI);
                cairo_set_source_rgba(CR, ERROR.r, ERROR.g, ERROR.b, 0.28);
                cairo_stroke(CR);
                const double F = std::clamp(static_cast<double>(charge), 0.0, 1.0);
                if (F > 0.0) {
                    cairo_set_line_cap(CR, CAIRO_LINE_CAP_ROUND);
                    cairo_arc(CR, 0, 0, 17, -PI / 2, -PI / 2 + F * 2 * PI);
                    source(CR, ERROR);
                    cairo_stroke(CR);
                }
                dot(ERROR);
                break;
            }
            case EMark::None: break;
        }
        cairo_destroy(CR);

        // drop-shadow(0 0 1.5px rgba(8,9,10,.95)): a Gaussian of 0.75 px.
        std::vector<float> A;
        if (shadow) {
            A = alphaOf(S);
            gaussian(A, SIZE, SIZE, 0.75 * s);
        }
        auto I = finish(S, shadow ? &A : nullptr, {8 / 255.0, 9 / 255.0, 10 / 255.0, 0.95}, opacity);
        cairo_surface_destroy(S);
        I.ax = I.ay = static_cast<float>(C);
        return I;
    }

    SImage paintLabel(const SLabel& L, float scale) {
        const double s = std::max(0.5f, scale);
        using E        = SLabel::EStyle;

        // The texts first, to know the panel's size.
        std::vector<SText> row1;
        SText              second;
        bool               hasSecond = false;
        double             padX = 14, padY = 10;
        bool               hasShadow = true;
        double             opacity   = 1.0;
        switch (L.style) {
            case E::Full:
                row1.push_back(text(L.name, 800, 16, HEADING, s, SANS, 300));
                if (!L.note.empty()) {
                    row1.push_back(text("·", 800, 16, DOT, s));
                    row1.push_back(text(L.note, 600, 15, SUB, s, SANS, 440));
                }
                break;
            case E::Compact:
                row1.push_back(text(L.name, 800, 15, HEADING, s, SANS, 300));
                padX = 12, padY = 7, hasShadow = false, opacity = 0.8;
                break;
            case E::Alert:
                row1.push_back(text(L.name, 800, 16, ERROR, s, SANS, 520));
                if (!L.note.empty()) {
                    second    = text(L.note, 600, 15, SUB, s, SANS, 520);
                    hasSecond = true;
                }
                break;
            case E::Quiet:
                row1.push_back(text(L.name, 600, 15, SUB, s, SANS, 520));
                padX = 12, padY = 8, hasShadow = false;
                break;
            case E::Plate:
                // The draft's plate: padding 8/14, radius 10, 800 15.
                row1.push_back(text(L.name, 800, 15, HEADING, s, SANS, 300));
                padX = 14, padY = 8, hasShadow = false;
                break;
            case E::Toast:
                row1.push_back(text(L.name, 800, 18, HEADING, s, SANS, 640));
                second    = text(L.note, 600, 15, SUB, s, SANS, 640);
                hasSecond = !L.note.empty();
                padX = 24, padY = 18;
                break;
        }

        // Row 1 sits on one baseline, 8 px apart.
        double ascent = 0, descent = 0, row1W = 0;
        for (size_t i = 0; i < row1.size(); ++i) {
            ascent  = std::max(ascent, row1[i].baseline);
            descent = std::max(descent, row1[i].h - row1[i].baseline);
            row1W += row1[i].w + (i ? 8 * s : 0);
        }
        const double ROW1_H = ascent + descent;

        // The keys: groups 20 apart, caps and their label 6 apart.
        struct SGroup {
            std::vector<SText> caps;
            SText              label;
            double             w = 0;
            float              alpha = 1.f;
        };
        std::vector<SGroup> groups;
        double              keysW = 0;
        if (L.style == E::Full)
            for (const auto& K : L.keys) {
                SGroup G;
                G.alpha = K.alpha;
                for (const auto& C : K.caps) {
                    G.caps.push_back(text(C, CAP_SMALL.weight, CAP_SMALL.px, CODE, s));
                    G.w += capWidth(G.caps.back(), CAP_SMALL, s) + 6 * s;
                }
                G.label = text(K.label, 700, 14, HINT, s);
                G.w += G.label.w;
                keysW += G.w + (groups.empty() ? 0 : 20 * s);
                groups.push_back(std::move(G));
            }

        const double GAP = (hasSecond ? L.gap : 10) * s;
        double       contentW = std::max(row1W, keysW), contentH = ROW1_H;
        if (hasSecond)
            contentW = std::max(contentW, second.w), contentH += GAP + second.h;
        if (!groups.empty())
            contentH += 10 * s + CAP_SMALL.h * s;
        const double PW = std::ceil(contentW + 2 * padX * s), PH = std::ceil(contentH + 2 * padY * s);
        const double R  = (L.style == E::Toast ? 22 : L.style == E::Plate ? 10 : 14) * s;

        const SMargins M = hasShadow ? floatMargins(s) : SMargins{2 * s, 2 * s, 2 * s, 2 * s};
        const int      W = static_cast<int>(std::ceil(PW + M.l + M.r)), H = static_cast<int>(std::ceil(PH + M.t + M.b));
        auto*          S  = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, W, H);
        auto*          CR = cairo_create(S);
        const double   X0 = std::round(M.l), Y0 = std::round(M.t);
        drawPanel(CR, X0, Y0, PW, PH, R, s);

        double x = X0 + padX * s;
        double y = Y0 + padY * s;
        // Row 1 centred in Compact and Quiet (one text), left otherwise.
        for (size_t i = 0; i < row1.size(); ++i) {
            draw(CR, row1[i], x, y + ascent - row1[i].baseline);
            x += row1[i].w + 8 * s;
        }
        y += ROW1_H;
        if (hasSecond) {
            y += GAP;
            draw(CR, second, X0 + padX * s, y);
            y += second.h;
        }
        if (!groups.empty()) {
            y += 10 * s;
            x = X0 + padX * s;
            for (const auto& G : groups) {
                cairo_push_group(CR);
                double gx = x;
                for (const auto& C : G.caps) {
                    drawCap(CR, C, gx, y, CAP_SMALL, s);
                    gx += capWidth(C, CAP_SMALL, s) + 6 * s;
                }
                draw(CR, G.label, gx, y + (CAP_SMALL.h * s - G.label.h) / 2);
                cairo_pop_group_to_source(CR);
                cairo_paint_with_alpha(CR, G.alpha);
                x += G.w + 20 * s;
            }
        }
        cairo_destroy(CR);

        std::vector<float> A;
        if (hasShadow) {
            A = floatShadow(W, H, X0, Y0, PW, PH, R, s);
        }
        auto I = finish(S, hasShadow ? &A : nullptr, {0, 0, 0, 0.5}, opacity);
        cairo_surface_destroy(S);
        I.ax = static_cast<float>(X0 + PW / 2);
        I.ay = static_cast<float>(Y0);
        return I;
    }

    SImage paintGunPill(float scale) {
        const double s = std::max(0.5f, scale);
        SText        name  = text("Process gun", 800, 16, TEXT, s);
        SText        note  = text("click closes · hold 1 s kills", 600, 15, SUB, s);
        SText        cap   = text("F7", CAP.weight, CAP.px, CODE, s);
        SText        hint  = text("Put away", 700, 15, HINT, s);
        const double CAPW  = capWidth(cap, CAP, s);
        const double PW    = std::ceil((20 + 8 + 14) * s + name.w + 14 * s + note.w + (14 + 10) * s + CAPW + 6 * s + hint.w + 6 * s + 10 * s);
        const double PH    = 52 * s;
        const double R     = 22 * s;
        const SMargins M   = floatMargins(s);
        const int      W   = static_cast<int>(std::ceil(PW + M.l + M.r)), H = static_cast<int>(std::ceil(PH + M.t + M.b));
        auto*          S   = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, W, H);
        auto*          CR  = cairo_create(S);
        const double   X0 = std::round(M.l), Y0 = std::round(M.t);
        drawPanel(CR, X0, Y0, PW, PH, R, s);
        const double MID = Y0 + PH / 2;
        double       x   = X0 + 20 * s;
        cairo_arc(CR, x + 4 * s, MID, 4 * s, 0, 2 * std::numbers::pi);
        source(CR, ERROR);
        cairo_fill(CR);
        x += (8 + 14) * s;
        // The two texts on one baseline, centred as a line.
        const double ASC = std::max(name.baseline, note.baseline);
        const double LH  = ASC + std::max(name.h - name.baseline, note.h - note.baseline);
        const double TY  = MID - LH / 2;
        draw(CR, name, x, TY + ASC - name.baseline);
        x += name.w + 14 * s;
        draw(CR, note, x, TY + ASC - note.baseline);
        x += note.w + (14 + 10) * s;
        drawCap(CR, cap, x, MID - CAP.h * s / 2, CAP, s);
        x += CAPW + 6 * s;
        draw(CR, hint, x, MID - hint.h / 2);
        cairo_destroy(CR);
        auto A = floatShadow(W, H, X0, Y0, PW, PH, R, s);
        auto I = finish(S, &A, {0, 0, 0, 0.5}, 1.0);
        cairo_surface_destroy(S);
        I.ax = static_cast<float>(X0 + PW / 2);
        I.ay = static_cast<float>(Y0);
        return I;
    }
    SImage paintHintPill(const std::vector<SKey>& keys, float scale) {
        const double s = std::max(0.5f, scale);
        struct SGroup {
            std::vector<SText> caps;
            SText              label;
            double             w = 0;
        };
        std::vector<SGroup> groups;
        double              inner = 0;
        for (const auto& K : keys) {
            SGroup G;
            for (const auto& C : K.caps) {
                G.caps.push_back(text(C, CAP.weight, CAP.px, CODE, s));
                G.w += capWidth(G.caps.back(), CAP, s) + 8 * s;
            }
            G.label = text(K.label, 700, 15, HINT, s);
            G.w += 4 * s + G.label.w;
            inner += G.w + (groups.empty() ? 0 : 30 * s);
            groups.push_back(std::move(G));
        }
        const double   PW = std::ceil(inner + 40 * s), PH = 52 * s, R = 22 * s;
        const SMargins M  = floatMargins(s);
        const int      W = static_cast<int>(std::ceil(PW + M.l + M.r)), H = static_cast<int>(std::ceil(PH + M.t + M.b));
        auto*          S  = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, W, H);
        auto*          CR = cairo_create(S);
        const double   X0 = std::round(M.l), Y0 = std::round(M.t);
        drawPanel(CR, X0, Y0, PW, PH, R, s);
        const double MID = Y0 + PH / 2;
        double       x   = X0 + 20 * s;
        for (const auto& G : groups) {
            for (const auto& C : G.caps) {
                drawCap(CR, C, x, MID - CAP.h * s / 2, CAP, s);
                x += capWidth(C, CAP, s) + 8 * s;
            }
            x += 4 * s;
            draw(CR, G.label, x, MID - G.label.h / 2);
            x += G.label.w + 30 * s;
        }
        cairo_destroy(CR);
        auto A = floatShadow(W, H, X0, Y0, PW, PH, R, s);
        auto I = finish(S, &A, {0, 0, 0, 0.5}, 1.0);
        cairo_surface_destroy(S);
        I.ax = static_cast<float>(X0 + PW / 2);
        I.ay = static_cast<float>(Y0);
        return I;
    }

    SImage paintArrow(float scale) {
        // The draft's path, 22 x 30 at scale 1, white with a 1.6 px dark
        // edge, so it reads on light and dark content; a shadow
        // 0 2px 4px rgba(0,0,0,.5) under it.
        const double s    = std::max(0.5f, scale);
        const double PAD  = 8 * s;
        const int    W    = static_cast<int>(std::ceil(22 * s + 2 * PAD)), H = static_cast<int>(std::ceil(30 * s + 2 * PAD));
        auto*        S    = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, W, H);
        auto*        CR   = cairo_create(S);
        cairo_translate(CR, PAD, PAD);
        cairo_scale(CR, s, s);
        static constexpr double P[][2] = {{1.5, 1.5}, {1.5, 24}, {7.2, 18.6}, {11, 27.6}, {15, 25.9}, {11.3, 17.1}, {19, 17.1}};
        cairo_move_to(CR, P[0][0], P[0][1]);
        for (size_t i = 1; i < std::size(P); ++i)
            cairo_line_to(CR, P[i][0], P[i][1]);
        cairo_close_path(CR);
        source(CR, TITLE);
        cairo_fill_preserve(CR);
        cairo_set_line_join(CR, CAIRO_LINE_JOIN_ROUND);
        cairo_set_line_width(CR, 1.6);
        cairo_set_source_rgb(CR, 8 / 255.0, 9 / 255.0, 10 / 255.0);
        cairo_stroke(CR);
        cairo_destroy(CR);
        // The shadow: the arrow's alpha, 2 px down, a Gaussian of 2 px.
        auto       A = alphaOf(S);
        const auto SH = static_cast<int>(std::lround(2 * s));
        for (int y = H - 1; y >= 0; --y)
            for (int x = 0; x < W; ++x)
                A[static_cast<size_t>(y) * W + x] = y >= SH ? A[static_cast<size_t>(y - SH) * W + x] : 0.f;
        gaussian(A, W, H, 2 * s);
        auto I = finish(S, &A, {0, 0, 0, 0.5}, 1.0);
        cairo_surface_destroy(S);
        I.ax = static_cast<float>(PAD + 1.5 * s);
        I.ay = static_cast<float>(PAD + 1.5 * s);
        return I;
    }

    SImage paintPointerRing(float scale) {
        // 44 px: a 2 px line inside its edge, rgba(92,184,230,.55), and a
        // glow 0 0 18px rgba(92,184,230,.35) outside it.
        const double s    = std::max(0.5f, scale);
        const int    SIZE = static_cast<int>(std::ceil((44 + 2 * 20) * s / 2)) * 2;
        auto*        S    = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, SIZE, SIZE);
        auto*        CR   = cairo_create(S);
        const double C    = SIZE / 2.0, R = 22 * s;
        cairo_arc(CR, C, C, R, 0, 2 * std::numbers::pi);
        cairo_set_source_rgba(CR, 0, 0, 0, 1);
        cairo_fill(CR);
        auto GLOW = alphaOf(S);
        gaussian(GLOW, SIZE, SIZE, 9 * s);
        const auto DISC = alphaOf(S);
        cairo_set_operator(CR, CAIRO_OPERATOR_SOURCE);
        cairo_set_source_rgba(CR, 0, 0, 0, 0);
        cairo_paint(CR);
        cairo_set_operator(CR, CAIRO_OPERATOR_OVER);
        cairo_arc(CR, C, C, R - s, 0, 2 * std::numbers::pi);
        cairo_set_line_width(CR, 2 * s);
        cairo_set_source_rgba(CR, UI.r, UI.g, UI.b, 0.55);
        cairo_stroke(CR);
        cairo_destroy(CR);
        for (size_t i = 0; i < GLOW.size(); ++i)
            GLOW[i] *= 1.0f - DISC[i];
        auto I = finish(S, &GLOW, {UI.r, UI.g, UI.b, 0.35}, 1.0);
        cairo_surface_destroy(S);
        I.ax = I.ay = static_cast<float>(C);
        return I;
    }
    SImage paintRoomCheck(const SRoomCheck& check, float scale) {
        const double s = std::max(0.5f, scale);
        const double PW = 400 * s, PADX = 24 * s, PADY = 22 * s, GAP = 18 * s, COL = 108 * s, ROWGAP = 12 * s;
        SText        head  = text("ROOM CHECK", 800, 14, UI, s, SANS, 0, 14 * 0.08);
        SText        world = text(check.world, 400, 13, LABEL, s, MONO, 160);
        std::vector<std::pair<SText, SText>> rows;
        for (const auto& [L, V] : check.rows)
            rows.emplace_back(text(L, 700, 15, LABEL, s), text(V, 500, 15, VALUE, s, MONO, (400 - 48 - 108)));
        SText collision = text("Collision", 600, 15, SUB, s), body = text("Your body", 600, 15, SUB, s);
        SText cap = text("F3", CAP.weight, CAP.px, CODE, s), hide = text("Hide", 700, 15, HINT, s);

        double rowsH = 0;
        for (const auto& [L, V] : rows)
            rowsH += std::max(L.h, V.h);
        rowsH += ROWGAP * std::max<double>(0, rows.size() - 1.0);
        const double HEAD_H = std::max(head.h, world.h), LEGEND_H = std::max(collision.h, body.h);
        const double PH = std::ceil(PADY * 2 + HEAD_H + GAP + rowsH + GAP + s + GAP + LEGEND_H + GAP + CAP.h * s);
        const double R  = 22 * s;

        const SMargins M  = floatMargins(s);
        const int      W  = static_cast<int>(std::ceil(PW + M.l + M.r)), H = static_cast<int>(std::ceil(PH + M.t + M.b));
        auto*          S  = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, W, H);
        auto*          CR = cairo_create(S);
        const double   X0 = std::round(M.l), Y0 = std::round(M.t);
        drawPanel(CR, X0, Y0, PW, PH, R, s);

        double       y = Y0 + PADY;
        const double X = X0 + PADX;
        // The header: a dot, ROOM CHECK, and the world on the right.
        cairo_arc(CR, X + 4 * s, y + HEAD_H / 2, 4 * s, 0, 2 * std::numbers::pi);
        source(CR, UI);
        cairo_fill(CR);
        draw(CR, head, X + 18 * s, y + (HEAD_H - head.h) / 2);
        draw(CR, world, X0 + PW - PADX - world.w, y + (HEAD_H - world.h) / 2);
        y += HEAD_H + GAP;
        // The rows on their baselines.
        for (const auto& [L, V] : rows) {
            const double ASC = std::max(L.baseline, V.baseline);
            draw(CR, L, X, y + ASC - L.baseline);
            draw(CR, V, X + COL, y + ASC - V.baseline);
            y += std::max(L.h, V.h) + ROWGAP;
        }
        y += GAP - ROWGAP;
        cairo_rectangle(CR, X, y, PW - 2 * PADX, s);
        source(CR, LINE);
        cairo_fill(CR);
        y += s + GAP;
        // The legend: the colours F3 draws -- the collision triangles
        // (MapModel's debug shader) and the player's capsule.
        double x = X;
        for (const auto& [T, C] : {std::pair<const SText*, SRGBA>{&collision, {1.0, 0.1, 0.1, 1.0}}, {&body, {0.2, 1.0, 0.3, 1.0}}}) {
            roundRect(CR, x, y + LEGEND_H / 2 - 1.5 * s, 18 * s, 3 * s, 2 * s);
            source(CR, C);
            cairo_fill(CR);
            draw(CR, *T, x + 26 * s, y + (LEGEND_H - T->h) / 2);
            x += 26 * s + T->w + 24 * s;
        }
        y += LEGEND_H + GAP;
        drawCap(CR, cap, X, y, CAP, s);
        draw(CR, hide, X + capWidth(cap, CAP, s) + 12 * s, y + (CAP.h * s - hide.h) / 2);
        cairo_destroy(CR);

        auto A = floatShadow(W, H, X0, Y0, PW, PH, R, s);
        auto I = finish(S, &A, {0, 0, 0, 0.5}, 1.0);
        cairo_surface_destroy(S);
        I.ax = static_cast<float>(X0);
        I.ay = static_cast<float>(Y0);
        return I;
    }
    SImage paintTag(const std::string& str, float scale) {
        // 30 high, 10 either side, radius 9, a panel with its line; the
        // text 400 13 mono in Larch's blue.
        const double s  = std::max(0.5f, scale);
        SText        t  = text(str, 400, 13, UI, s, MONO);
        const double PW = std::ceil(t.w + 20 * s), PH = 30 * s;
        const int    W = static_cast<int>(PW + 4), H = static_cast<int>(std::ceil(PH) + 4);
        auto*        S  = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, W, H);
        auto*        CR = cairo_create(S);
        drawPanel(CR, 2, 2, PW, PH, 9 * s, s);
        draw(CR, t, 2 + 10 * s, 2 + (PH - t.h) / 2);
        cairo_destroy(CR);
        auto I = finish(S, nullptr, {}, 1.0);
        cairo_surface_destroy(S);
        I.ax = I.ay = 2.0f;
        return I;
    }
}
