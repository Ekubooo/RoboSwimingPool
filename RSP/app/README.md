# Application and playback

`application.py` owns the main window and the selection-to-session transition.
`playback.py` binds session callbacks and advances the policy/environment through
the session interface; `ui/` owns panels, shortcuts and queued CPU commands.
The same playback binding serves the legacy MimicKit entry point. Unbinding
removes its callbacks without closing the borrowed environment or window.

Scene assembly lives in `simulation/scenes/assembly.py`; UI panels own selection
and draft state, not a second placement implementation. Asset preparation and
scene-placement errors are handled separately by the application.

## Scene selection and startup

Custom assembly includes **Generate terrain**, a modal inside the existing
Newton window. Choose Stairs, Random grid or Gap, adjust the small parameter
set, preview, and save under a new name. A successful save immediately adds and
selects the container card; reopening the application discovers it again.
Grid height is seeded, but OBJ placement always uses the first cell. Central
models remain an assembly choice and are not stored in the terrain recipe.

The editor uses the same orthographic camera and renderer as asset cards.
Draft generation and texture replacement run outside ImGui drawing and only
when parameters change. No SDF, volume map or CoACD is run by preview/save.
Escape cancels the draft; window close cancels selection. Unconfirmed drafts
do not write source assets, and all preview textures belong to selection.

`python tools/run_rsp.py` opens the Newton window and reads scene choices from
the source catalog in `data/assets_RSP/origin`. Before selection, every discovered
OBJ is validated and prepared with a collision model and SDF. This also works
with an empty runtime directory. Failures are displayed and excluded. Presets
show asset compositions; custom mode controls prop quantities, placement and
the optional static model of procedural terrain containers. Asset references identify
an asset by category and name.

- `RSP/rendering/preview/geometry.py` reads authored robot geometry on the CPU
  through Newton's MJCF importer, uses the source primitive dimensions for
  primitives, reads prepared OBJ meshes, and draws the reference bunny or terrain.
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
