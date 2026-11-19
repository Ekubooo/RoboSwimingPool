# Application and playback

`application.py` owns the main window and the selection-to-session transition.
`playback.py` binds session callbacks and advances the policy/environment through
the session interface; `ui/` owns panels, shortcuts and queued CPU commands.
The same playback binding serves the legacy MimicKit entry point. Unbinding
removes its callbacks without closing the borrowed environment or window.

## Scene selection and startup

`python tools/run_rsp.py` opens the Newton window and reads scene choices from
the source catalog in `data/assets_RSP/origin`. The selector works with an
empty runtime directory. Presets show a composition of the source assets, and
custom mode controls prop quantities and placement. Asset references identify
an asset by category and name.

- `RSP/rendering/preview/geometry.py` reads authored robot geometry on the CPU
  through Newton's MJCF importer, uses the source primitive dimensions for
  props, and draws the container and its positioned bunny mesh.
- `ui/cards.py` draws cards and queues preview requests. `application.select_scene`
  processes one visible-card request per frame outside the draw callback.
- `RSP/rendering/preview/textures.py` renders orthographic thumbnails into GL
  textures owned by the selection panel.

The camera looks from +X/-Y/+Z (right-front azimuth 45 degrees, elevation 45
degrees). Thumbnails show the model's authored default pose, not a sampled GSI
spawn. G1 retains authored display colors; humanoid and props use demo colors.

Temporary geometry, buffers, shader and FBO objects are released after each
render. Start, Cancel, Escape, window close and setup failure remove callbacks
and release preview textures before simulation construction or context
destruction. A failed thumbnail displays a placeholder.

**Start simulation** submits the selection. The startup adapter checks the
selected assets against the support radius derived from the active fluid
settings. The [asset pipeline](../sim_ready/README.md#active-call-sequence-and-interfaces)
generates the missing or stale runtime assets before the simulation is
constructed. The same Newton window shows progress and then hosts playback.
A preparation error returns to the same selection panel with the error shown.
Closing the window during preparation stops startup at the next progress
boundary.

Focused checks (active conda environment, project root):

```powershell
python -B -m unittest discover -s tests -p "test_rsp_launcher_ui.py"
python -B -m unittest discover -s tests -p "test_rsp_playback.py"
```

These CPU contracts cover playback scheduling and window/callback ownership.
They do not open a real GL window. Interactive layout acceptance is a separate
check using `python tools/run_rsp.py`; see [local-check policy](../README.md#running-and-checking).
