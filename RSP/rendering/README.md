# RSP GPU rendering

## Optional fluid modes

The UI's **Fluid rendering** selector switches at the next frame boundary:

- **Original** (default): the unchanged opaque square billboard implementation.
- **Transparent**: square billboards with the original speed palette, weighted
  blended OIT and scene-depth intersection fading. Opacity increases smoothly
  from 0.15 at 2 m/s to 0.9 at 6.5 m/s, clamped outside this interval.
  Scene intersections still apply the existing soft depth fade.
- **SSFR**: sphere depth and thickness, separable bilateral depth smoothing,
  reconstructed normals, thickness absorption, Fresnel environment reflection
  and screen-space refraction. Surface radius is 1.5 times the physical radius;
  this never changes simulation geometry. The default refractive index is 1.333.

Within SSFR, **SSFR material** selects **Water** (default) or **Velocity**:

- Water defaults to the first SSFR implementation's deep-blue base color
  `(0.02, 0.12, 0.2)`, retaining thickness, refraction, reflection and highlights.
  **Water color** selects **Current** (this original water material) or
  **Billboard blue** (`(29,119,231)/255` with neutral thickness absorption to
  preserve the billboard blue hue).
  This selection can change while paused and persists across reset/mode changes.
- Velocity reuses the billboard's blue/light-blue/white palette. Coverage rises
  with `smoothstep(2, 6.5, speed)`; at 6.5 m/s a uniformly fast visible surface
  uses 90% white and 10% refracted background, overriding reflection/highlights.
  Mixed surface
  particles contribute a depth-weighted average, so an isolated fast particle
  does not whiten an entire layer.
  Its thickness attenuation uses the mean absorption coefficient equally for
  RGB; this avoids warm color shifts from Water's selective RGB absorption.
  Both optional modes use the original billboard's 2 / 6.5 m/s color thresholds.
  `alpha_high` in `fluid/config.py` controls the optional modes' white opacity;
  it does not affect Original or Water's thickness-based material.

Velocity adds color/weight/coverage attachments to the thickness pass, using
the existing packed GPU data. Only fragments within two display radii of the
nearest fluid surface contribute attributes; opaque scene depth also clips
them. Water skips this accumulation. Both presets share the unchanged depth,
thickness and smoothing algorithms. Switching is queued at the frame boundary,
works while paused, and preserves the selection across reset and mode changes.
No simulation Graph or physical parameters are changed.

Profile a preset with `python tools/profile_rsp_fluid_modes.py SSFR --ssfr-preset Water`
(or `Velocity`). Add `--check-materials` to perform a short
paused-switch/reset check after the timed batch. Preset measurements use a
separate `output/rsp_render_profile/ssfr_presets/` directory so previous mode
results remain available for historical comparison.
See `docs/ssfr_material_presets_results.md` for the measured preset comparison.

Optional implementations live in `fluid/`; edit `fluid/config.py` for their
startup settings. They share a private CUDA–GL particle buffer, borrowing the
legacy packing/resource implementation without changing its source. Only the
selected mode uploads particles and draws. New modes lazily allocate shaders
and textures, so their first use can pause briefly; resources are reused until
resize/model replacement/close. Scene reset preserves the selected mode.

`FluidPipeline` wraps this viewer instance's scene render call: Newton first
draws the scene and resolves MSAA, then the new mode samples that color/depth
into separate targets and commits the result before UI/display-frame capture.
Original takes the unchanged draw path with no optional framebuffer allocation.
No solver Graph is recaptured and no particle state is read back. Model changes
and shutdown release CUDA registration before GL storage. An initialization
failure keeps the previous mode and reports the error in the panel.

SSFR approximates only the visible surface; refraction is limited to the current
screen. No foam, spray, anisotropy, caustics or multiple optical bounces are added.
The first version uses single-sample fluid targets over the resolved MSAA scene.

Necessary checks: `tests/test_rsp_fluid_rendering.py`; per-mode performance tool:
`python tools/profile_rsp_fluid_modes.py Original` (or `Transparent` / `SSFR`).
The tool warms 30 frames and samples 180 once, with GPU timestamps resolved
after the batch. Nested GPU timings are attribution, not additive CPU costs.
It produces one still image per mode, with no recording tests or video files.
Use `--check-switches` for the short paused-switch/reset check instead of timing
another batch. Measurements and limitations are in
`docs/rendering_fluid_modes_results.md` at the repository root.

`BillboardBatch` owns one CUDA/OpenGL instance buffer, a quad and its shaders.
`NewtonBillboardViewer` adapts that batch to Newton 1.7's existing scene, window,
camera and UI. MimicKit's `PBFViewer` name remains the application entry point;
the same viewer is used by the headless video recorder. `RigidBatch` also owns
one shared CUDA/OpenGL buffer for all rigid instances in a layer.
Right-drag pulls a rigid body using Newton's original picking forces. The
selection-only model in `picking.py` excludes collision-only geometry so the
invisible pool walls cannot intercept the ray. Visible static geometry still
occludes selection; the complete collision model is unchanged. Mouse picking
and its helper line retain Newton's interaction-time readbacks.

Importing the backend does not initialize OpenGL; a context is needed only when
constructing a viewer, so non-visual simulation retains its display-free path.

`PlaybackGui` composes the existing Newton GUI, sharing its ImGui context,
input handling and GL backend. The playback panel only builds run/reset,
camera, geometry/particle visibility and application controls. Model values,
gravity, selection attributes, plots and diagnostic panels are not read or
built each frame. Side/free UI callback registration remains available for
future controls. Picking and explicit frame capture retain their on-demand
readback; normal idle playback has no simulation-state readback in this UI.
The scene adapter no longer submits the static pool outline.

## Data and rendering

Call `update(positions, velocities, radii)` with contiguous, one-dimensional
Warp arrays on the batch's CUDA device (`vec3`, `vec3`, `float32`). The arrays
are borrowed and never modified. A kernel writes two `vec4` values per particle:
center/radius and RGBA. The GL buffer is registered once per particle count and
updated through `wp.RegisteredGLBuffer`, without CPU particle readback. Interop
failure raises an error instead of silently switching to host copies.

Call `draw(view, projection)` with OpenGL column-major camera matrices. The
vertex shader expands each square in camera space, including during paused
redraws. The caller owns the current context, framebuffer and viewport. Drawing
restores the GL state it changes. Call `close()` before destroying the context;
it releases CUDA registration before deleting the GL buffers and is idempotent.

The visual defaults follow the active WarpLab billboard implementation:

| Speed | Color |
| --- | --- |
| 0 | `#1D77E7` |
| 2.0 | `#A0DDFF` |
| 6.5 and above | `#FFFFFF` |

Colors interpolate linearly in shader values. These are **display thresholds**,
configurable using `speed_color_mid` and `speed_color_max` on the viewer or batch;
they do not change the solver's velocity limits. Squares have side length twice
the model particle radius, opaque flat color, depth testing and depth writes.
There is no particle lighting, shadow pass, edge overlay or transparency sort.

## Newton boundary

Version-specific integration is confined to `newton_gl.py`: the scene-object
`render()`/`destroy()` protocol, instance attributes, shape batching, camera
matrices and layer-qualified names. Particles draw in the opaque scene pass, sharing rigid
depth before transparency, MSAA resolve and ImGui. Human geometry, scene lines,
input callbacks and recording retain their Newton paths. No global monkeypatch
or modified installed package is needed.

Rigid records contain a transform, RGB/opacity and material in 96 bytes. A single
GPU kernel writes the shared GL buffer; group VAOs address its ranges and reuse
Newton's mesh shaders. Capsules have persistent cylinder/end-cap instances.
Colors follow the model's GPU array every frame. After changing opacity across
opaque/transparent passes, call `invalidate_rigid_appearance()` explicitly;
ordinary frames do not read opacity flags back to the CPU. Optional diagnostic
overlays and picking retain Newton's paths and may perform their own readbacks.

RSP adapts only its renderer instance to nonblocking event polling. The existing
application limiter still controls playback pacing; window/UI events continue
to be dispatched. Hidden COM markers skip Newton's otherwise unconditional
update/upload. Enabling the COM overlay still uses Newton's original drawing.

Newton owns model/layer cleanup and calls the object's `destroy()`. The adapter
also handles the window-close event before the context disappears. When updating
Newton, run the rendering tests to verify this internal scene-object contract.

## Verification and timing

Use the project's WarpSim environment from the repository root. Select only the
checks relevant to the change; this list is not a required full suite. The
integration check covers policy loading and GSI reset with recording disabled.
Do not run recording tests or generate recording test artifacts.

```text
python -m unittest discover -s tests -p test_rsp_rendering.py -v
python -m unittest discover -s tests -p test_newton_pbf_refactor.py -v
python tools/check_newton_pbf_refactor.py integration --revision after
python tools/profile_rsp_render.py --label billboard --hidden
```

The profiler runs the existing SMP policy and PBF scene with the real Newton UI.
`--hidden` hides the window after creation; it does not use Newton's headless
mode, which disables UI. It warms up 30 frames, then measures 180 frames in each
of two modes: normal rate limiting and a process-local bypass of that limiter.
Simulation frequencies, production pacing and Presentation are unchanged.

Reports and screenshots go to `output/rsp_render_profile/<label>`. Frame times
include policy, physics, scene/UI rendering and presentation. Render timings are
CPU call durations, including driver/interop waits, **not GPU kernel timings**.
GPU work is drained only at batch boundaries, with an additional completed-batch
mean reported. Screenshot readback, loading, JIT compilation and reset/warmup
are outside the sampled interval. Compare identical arguments and scene configs
before and after a change; unrelated WarpLab scenes are not an FPS baseline.
