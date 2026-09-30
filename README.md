# Hypr3D =^..^=

![alt text](images/Screenshot.png)

**A new perspective on window management -- literally.**
A Hyprland plugin that turns your workspace into a walkable 3D space where
windows float in mid-air, ready to be grabbed, dragged and resized.

> Experimental, pinned to Hyprland 0.56.2.

> It's kinda buggy, but I'm working on it.
## Installation
### Hyprpm

```bash
hyprpm add https://github.com/samine825/Hypr3D
hyprpm enable Hypr3D
```

(`hyprpm update` picks up new commits.)


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
hl.bind("SUPER + F12", hl.plugin.hypr3d.toggle)
```


## Controls

| Input                         | Action                                                   |
| -------------------------------| ----------------------------------------------------------|
| Mouse move                    | Look around                                              |
| WASD                          | Move                                                     |
| Shift / Space                 | Move down / up                                           |
| Ctrl                          | Sprint                                                   |
| Super + Left click            | Drag window                                              |
| Super + Right click           | Resize window                                            |
| Super + Mouse wheel click     | rotate window                                            |
| Super + Mouse wheel scrolling | Zoom in on or zoom out from a window under the crosshair |
| Super + Left Alt              | Toggle keyboard mode (movement / window input)           |

## Configuration
### Lua config example:

```lua
if hl.plugin.hypr3d then
    -- binds
    hl.bind("SUPER + F12", hl.plugin.hypr3d.toggle)
    -- parameters
    hl.plugin.hypr3d.config({
        panorama = "/home/samine/Downloads/Qwen_image_2.1_00048.png", -- 360° room background image (equirectangular)
        look_inertia = 0.03, -- mouse-look glide after the mouse stops, seconds (0 = off)
        move_inertia = 0.05, -- WASD glide after the keys are released, seconds (0 = off)
        move_speed = 4.0, -- walking speed, world m/s (sprint is ×2.5 this value)
        sensitivity = 0.0025, -- mouse look sensitivity, radians per pointer count
        window_scale = 0.5, -- window size multiplier: real pixel size at a base density of 100 px/m, times this
        spawn_distance = 5, -- distance from the camera where new windows spawn
        player_spawn = { x = 0, y = 0, z = 0 }, -- player spawn point in the room

        -- Optional glTF 2.0 map (.glb / .gltf): meshes, textures and player
        -- collision. Nodes named "nocol*" render but never collide.
        -- map = {
        --     path = "/path/to/map.glb",   -- omit or "" for the grid room
        --     position = { x = 0, y = 0, z = 0 },
        --     rotation = { x = 0, y = 0, z = 0 }, -- degrees, XYZ
        --     scale = 1.0,
        --     debug_collision = false, -- red x-ray wireframe of collision tris
        -- },
        -- That's all for now :p
    })
end
```