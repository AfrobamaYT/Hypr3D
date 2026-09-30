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
hl.bind("SUPER + F12", hl.plugin.hypr3d.toggle)
hl.config({
    plugin = {
        hypr3d = {
            -- U can set a panorama pic:
            panorama = "/home/samine/Downloads/Qwen_image_2.1_00048.png",
            -- And other world parameters:
            look_inertia = 0.03,
            move_inertia = 0.05,
            move_speed = 4.0, -- Walking speed, world m/s (default 8.0, range 0.5–50). Sprinting is ×2.5 this value
            sensitivity = 0.0025, -- Mouse sensitivity, radians per pointer count (default 0.0025, range 0.0001–0.05)
            window_scale = 0.5, -- Window scale relative to the world: a multiplier applied to their actual pixel size at a base density of 100 px/m (default 1.0, range 0.1–8)
            spawn_distance = 5, -- distance from the camera to the spawn point for the new window (default 10.0, range 1–100)
            -- That's all for now :p
        },
    },
})
end
```