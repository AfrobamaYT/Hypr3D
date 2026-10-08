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
        Resize,  // brackets with a diagonal stroke: Super + right drag resizes from this corner
        Lock,    // amber brackets round an amber block: this window has a fixed size
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
            Toast,   // a note at the top: a heading, a line under it
            Plate,   // a name on a small plate, as under a portal
        };
        EStyle            style = EStyle::Full;
        std::string       name, note;
        std::vector<SKey> keys;
        float             gap = 10.0f; // Alert: between the two lines
    };

    SImage paintLabel(const SLabel& label, float scale);

    // The process gun's mode pill, 48 px from the top while F7 is on.
    SImage paintGunPill(float scale);

    // F3's room check: the world's name, rows of a label and a value, and
    // the legend of the colours F3 draws in the room. The anchor is the
    // panel's top left.
    struct SRoomCheck {
        std::string                                      world;
        std::vector<std::pair<std::string, std::string>> rows;
    };
    SImage paintRoomCheck(const SRoomCheck& check, float scale);

    // A small mono tag, as F2's "1 : 1" over the window's corner. The
    // anchor is its top left.
    SImage paintTag(const std::string& text, float scale);

    // The size readout at a resized window's corner (the owner's approved
    // "Room Interactions" draft, 2026-10-08): a line of size in mono and a
    // grey line under it. The anchor is the panel's top left.
    enum class ETone : uint8_t {
        Accent, // drawing a size
        Amber,  // at a limit, or the app chose another size
        Muted,  // waiting for the app
    };
    SImage paintReadout(const std::string& text, const std::string& sub, ETone tone, float scale);

    // A pill of key hints, as at the bottom of Super+F's screen; `bare`:
    // the keys alone, no pill. Its anchor is its top centre, bare its top
    // left.
    SImage paintHintPill(const std::vector<SKey>& keys, float scale, bool bare = false);

    // Through a portal (the owner's approved "Portals" draft): the light of
    // the game's two colours over its blurred cover, w x h px, anchored at
    // its top left; the eyebrow and the game's name, anchored at the top
    // left; and the panel of its start -- or of its failure.
    struct SColour {
        double r = 0, g = 0, b = 0;
    };
    SImage paintPortalScrim(int w, int h, SColour ac, SColour ac2);
    SImage paintPortalTitle(const std::string& name, SColour ac, float scale);
    struct SStart {
        std::string title;        // "Starting DELTARUNE"
        int         step = 1, steps = 4;
        float       part = 0.45f; // of the current step
        std::string text, detail; // "Steam is starting the game", the command
        bool        failed = false; // title in red, text a sentence, no bars
    };
    SImage paintStartPanel(const SStart& start, float scale);

    // The pointer of a window in use: Larch's arrow, its anchor the tip; and
    // the ring that shows where it is, its anchor the centre.
    SImage paintArrow(float scale);
    SImage paintPointerRing(float scale);
}
