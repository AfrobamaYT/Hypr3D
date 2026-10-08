#pragma once

#include <cstdint>
#include <string>
#include <vector>

// The room's screen overlays -- the crosshair's marks, the label under it,
// the gun's mode pill -- painted with Cairo and Pango into pixels that
// GLScene draws as screen sprites. Measures from the owner's approved
// Claude Design draft (Room Overlays, 2026-10-08) at scale 1; everything is
// multiplied by the monitor's scale.
namespace Overlay {

    // Premultiplied RGBA, rows top down. (ax, ay) is the anchor inside the
    // image in pixels: the mark's centre, the label's top centre (the panel's
    // edge, not the shadow's), the pill's top centre.
    struct SImage {
        int                  w = 0, h = 0;
        float                ax = 0.0f, ay = 0.0f;
        std::vector<uint8_t> rgba;
        uint64_t             serial = 0; // new for every painted image
    };

    enum class EMark : uint8_t {
        None,
        Window,  // brackets: a window or a layer
        Carry,   // blue ring: an object Super+drag carries
        Fixed,   // white ring: an object fixed in place
        Portal,  // diamond
        Nothing, // dot: sky, floor
        Bin,     // red ring: letting go of the carried window closes it
        Gun,     // red cross: the gun on something it can shoot
        GunIdle, // white cross at 55 %: the gun on nothing with a process
        GunHold, // the ring filling while the kill is held
    };

    // charge: GunHold's fill, 0..1.
    SImage paintMark(EMark mark, float scale, float charge = 0.0f);

    struct SKey {
        std::vector<std::string> caps;  // the key caps, in order
        std::string              label; // what pressing them does
        float                    alpha = 1.0f;
    };

    struct SLabel {
        enum class EStyle : uint8_t {
            Full,    // name · note, a row of keys
            Compact, // the name only (after 2 s on the same target)
            Alert,   // name in red, the note under it
            Quiet,   // one line of grey text, no shadow
        };
        EStyle            style = EStyle::Full;
        std::string       name, note;
        std::vector<SKey> keys;
        float             gap = 10.0f; // Alert: between the two lines
    };

    SImage paintLabel(const SLabel& label, float scale);

    // The process gun's mode pill, 48 px from the top while F7 is on.
    SImage paintGunPill(float scale);

    // A pill of key hints, as at the bottom of Super+F's screen.
    SImage paintHintPill(const std::vector<SKey>& keys, float scale);

    // The pointer of a window in use: Larch's arrow, its anchor the tip; and
    // the ring that shows where it is, its anchor the centre.
    SImage paintArrow(float scale);
    SImage paintPointerRing(float scale);
}
