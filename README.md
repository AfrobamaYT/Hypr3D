# Hypr3D =^..^=

![alt text](images/Screenshot.png)

**A new perspective on window management -- literally.**
A Hyprland plugin that turns your workspace into a walkable 3D space

> Experimental, pinned to Hyprland 0.56.2.

## Installation
### Hyprpm

```bash
# Install latest
hyprpm add https://github.com/samine825/Hypr3D
hyprpm enable Hypr3D

# Update
hyprpm update
```



### Manual

```bash
# Build
cmake -S . -B build -DHYPRLAND_HEADERS=/var/cache/hyprpm/$USER/headersRoot
cmake --build build -j$(nproc)

# Load
hyprctl plugin load "$PWD/build/hypr3d.so"
```

# Use

To enable 3D:

```bash
hyprctl eval 'hl.plugin.hypr3d.toggle()'
```

Or bind in lua config:

```lua
if hl.plugin.hypr3d then
    hl.bind("SUPER + F12", hl.plugin.hypr3d.toggle)
    hl.plugin.hypr3d.config({
        world = {
            panorama = "~/Pictures/room.png", -- 360° background (equirectangular)
            grid = true,                      -- base grid platform at world zero
        },
        windows = {
            window_scale = 0.5,   -- window size multiplier (real px at 100 px/unit)
            spawn_distance = 5,   -- distance from the camera new windows spawn at
        },
        player = {
            look_sensitivity = 0.0025, -- mouse look, radians per pointer count
            look_inertia = 0.03,       -- look glide after the mouse stops, sec
            move_inertia = 0.05,       -- walk glide after keys release, sec
            move_speed = 4.0,          -- walking speed, world units/sec
            spawn = { x = 0, y = 0, z = 0 }, -- FEET position (eyes ride +1.62)
            flying = true,             -- false: gravity, Space jumps, Shift idle
        },
        scene = {                 -- unlimited named glTF objects
            map = {               -- (any names; "map" is convention only)
                path = "~/map.glb",
                transform = {
                    position = { x = 0, y = 0, z = 0 },
                    rotation = { x = 0, y = 0, z = 0 }, -- degrees, XYZ
                    scale    = { x = 1, y = 1, z = 1 }, -- per-axis
                },
                emissive_scale = 1.0, -- baked-light multiplier (0 = off)
                flat = true,          -- baked-map look (textures carry light)
                collision = true,     -- collide with the player
                static = true,        -- true: location geometry; false: the
                                      -- object is grabbable (Super+LMB carries
                                      -- it on the crosshair, release drops it)
                physics = false,      -- gravity + world collisions for the
                                      -- object (needs static = false)
            },
            -- prop = { path = "...", static = false, ... },
        },
    })
end
```

Every key is optional: missing keys keep their current value, wrong-typed
keys raise a lua error, and values are clamped to sane ranges on set.
Object names inside `scene` are free-form. The legacy `map = {...}` section
still works as a single-object scene (a `scene` table wins when both exist).

The grid platform spans 40x40 world units centered on zero; there is no
world floor clamp -- off its edge you fall (bring a map floor or fly).

## Controls

| Input                         | Action                                         |
| -------------------------------| ------------------------------------------------|
| Mouse move                    | Look around                                    |
| WASD                          | Move                                           |
| Space                         | Move up (flying) / Jump                        |
| Shift                         | Move down (flying)                             |
| Ctrl                          | Sprint                                         |
| Super + Left click            | Drag window                                    |
| Super + Right click           | Resize window                                  |
| Super + Mouse wheel click     | rotate window                                  |
| Super + Mouse wheel scrolling | Zoom window                                    |
| Super + Left Alt              | Toggle keyboard mode (movement / window input) |
| F3                            | Toggle debug HUD                               |

## Configuration
### Lua config example:

```lua
if hl.plugin.hypr3d then
    -- binds
    hl.bind("SUPER + F12", hl.plugin.hypr3d.toggle)
    -- parameters
    hl.plugin.hypr3d.config({
        -- every key is optional
        world = {
            panorama = "~/Pictures/room.png", -- 360° room background (equirectangular)
            grid = true,                      -- base 40x40 grid platform
        },
        windows = {
            window_scale = 0.5,   -- window size multiplier (real px at 100 px/m)
            spawn_distance = 5,   -- distance from the camera new windows spawn at
        },
        player = {
            look_sensitivity = 0.0025, -- mouse look, radians per pointer count
            look_inertia = 0.03,       -- look glide after the mouse stops, sec
            move_inertia = 0.05,       -- walk glide after keys release, sec
            move_speed = 4.0,          -- walking speed, m/s
            spawn = { x = 0, y = 0, z = 0 }, -- player position
            flying = true,             -- on alse: gravity, Space jumps off the ground, Shift does nothing
        },
        map = {
            path = "~/map.gltf",   -- glTF 2.0 map (.glb/.gltf)
            transform = {
                position = { x = 0, y = 0, z = 0 },
                rotation = { x = 0, y = 0, z = 0 },
                scale    = { x = 1, y = 1, z = 1 },
            },
            emissive_scale = 1.0, -- emission multiplier
            flat = true,          -- textures carry all lighting
            collision = true,     -- collide with the map
        },
    })
    -- That's all for now :p
end
```