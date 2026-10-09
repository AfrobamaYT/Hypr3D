# Desktop acceptance tests

- Run source checks with `git -C <checkout>`: build, rig and evidence directories can live beside the checkout.
- Use a dedicated offscreen compositor and verify its PID, configuration path and IPC socket before input or process-kill tests.
- Launch graphical fixtures through that compositor's IPC. Give GTK/browser fixtures a private D-Bus session and separate XDG configuration/data/cache directories; keep changing logs outside their test-data folder.
- Verify window/process state and recorded pixels. A successful build does not verify a desktop interaction.
- Use a fixed keymap for client shortcuts. Verify the client action; wtype's temporary layouts can produce correct room hotkeys while a GTK shortcut is missed.
- Request `hl.plugin.hypr3d.status()` before reading the diagnostic dump in an active room. The periodic file is throttled and can contain older pointer coordinates.
