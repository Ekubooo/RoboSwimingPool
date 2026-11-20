# RSP GPU rendering

## Optional fluid modes

The UI's **Fluid rendering** selector switches at the next frame boundary:

- **Original** (default): the unchanged opaque square billboard implementation.
- **Transparent**: square billboards with the original speed palette, weighted
  blended OIT and scene-depth occlusion, without intersection fading. Opacity
  increases smoothly from 0.8 at 2 m/s to 1.0 at 6.5 m/s, clamped outside this
  interval. `transparent_alpha_low/high` configure this mode independently.
- **SSFR**: sphere depth and thickness, separable bilateral depth smoothing,
  reconstructed normals, thickness absorption, Fresnel environment reflection
  and screen-space refraction. Surface radius is 1.5 times the physical radius;
  this never changes simulation geometry. The default refractive index is 1.333.

Transparent has two integer **opacity (%)** sliders: minimum 60–80% (default
80%) and maximum 80–100% (default 100%), each stepping by 1%. They affect only
this mode and are applied at the next frame boundary, including while paused.

**Transparent layer blending** is enabled by default and preserves weighted
OIT. Disabling it selects only the nearest square particle face per pixel,
then blends its speed color with the opaque scene once using its own opacity.
Rear particles contribute neither color nor opacity. A private depth buffer
selects the face without sorting or CPU readback; scene depth remains unchanged.
Speed colors, opacity sliders and automatic depth shading work in both settings.
The private target is allocated on first use, cached across switches and freed
on resize/close. Equally deep faces use the first drawn face; changing which
particle is nearest can expose square edges and cause color changes.
The prior fixed-scene comparison is recorded in
`docs/transparent_layer_results.md`, including results and limitations.

**Depth shading (automatic range)** enables near-bright/far-dark fluid color;
**Depth strength** ranges from 0 to 1, default 0.5. At the default strength,
the nearest visible fluid is multiplied by 1.05 and the farthest by 0.75, with
a smooth transition. Strength zero or disabling the effect restores unmodified
OIT color. White may clip when brightened. Neither opacity accumulation nor
background color is changed by this effect.

The particle pass accumulates weighted linear camera depth in an extra R32F
attachment. A GPU-only 2×2 min/max reduction of visible fluid pixels supplies
the current near/far range; no particle positions or depth limits are read to
the CPU. Empty and nearly flat depth ranges skip shading. Range textures are
cached and recreated on resize. Disabling the effect skips reduction passes.
No manual near/far distance controls are provided. The automatic range follows
camera/visibility changes and is not temporally filtered.

Compare the effect by switching depth shading in the playback UI with the same
80/100% opacity settings. See `docs/transparent_depth_results.md` for the prior
measured cost and check results.

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
  `alpha_high` in `fluid/config.py` controls SSFR Velocity's white opacity;
  it does not affect Original or Water's thickness-based material.

Velocity adds color/weight/coverage attachments to the thickness pass, using
the existing packed GPU data. Only fragments within two display radii of the
nearest fluid surface contribute attributes; opaque scene depth also clips
them. Water skips this accumulation. Both presets share the unchanged depth,
thickness and smoothing algorithms. Switching is queued at the frame boundary,
works while paused, and preserves the selection across reset and mode changes.
No simulation Graph or physical parameters are changed.

Switch Water/Velocity in the playback UI, including while paused and across
reset. See `docs/ssfr_material_presets_results.md` for the prior measured
preset comparison.

Optional implementations live in `fluid/`; edit `fluid/config.py` for their
startup settings. They compose the shared `ParticleInstances` resource and
`RegisteredBuffer` lifetime implementation used by billboards. Only the
selected mode uploads particles and draws. New modes lazily allocate shaders
and textures, so their first use can pause briefly; resources are reused until
resize/model replacement/close. Scene reset preserves the selected mode.

`NewtonBillboardViewer` installs the fluid pipeline around its scene render call: Newton first
draws the scene and resolves MSAA, then the new mode samples that color/depth
into separate targets and commits the result before UI/display-frame capture.
Original takes the unchanged draw path with no optional framebuffer allocation.
No solver Graph is recaptured and no particle state is read back. Model changes
and shutdown release CUDA registration before GL storage. An initialization
failure keeps the previous mode and reports the error in the panel.

SSFR approximates only the visible surface; refraction is limited to the current
screen. No foam, spray, anisotropy, caustics or multiple optical bounces are added.
The first version uses single-sample fluid targets over the resolved MSAA scene.

Prior mode measurements and limitations are in
`docs/rendering_fluid_modes_results.md`. The one-off performance scripts and
intermediate outputs were removed after acceptance. Historical commands in
those dated reports describe the original experiments, not maintained tools.

`BillboardBatch` owns its shaders and a `ParticleInstances` batch. Optional fluid
passes use the same quad and instance layout. `RegisteredBuffer` manages mapping,
unmapping and releasing the CUDA registration for both particle and rigid GL
buffers. `NewtonBillboardViewer` adapts these batches to Newton 1.7's scene,
window and camera. MimicKit retains its `PBFViewer` alias for compatibility.
Right-drag pulls a rigid body using Newton's original picking forces. The
selection-only model in `picking.py` excludes collision-only geometry so the
invisible pool walls cannot intercept the ray. Visible static geometry still
occludes selection; the complete collision model is unchanged. Mouse picking
and its helper line retain Newton's interaction-time readbacks.

Importing the backend does not initialize OpenGL; a context is needed only when
constructing a viewer, so non-visual simulation retains its display-free path.

The application installs `NewtonGuiHost` on the viewer for scene selection
and playback. The host shares Newton's ImGui context and handles frame callbacks
and input capture. `RSP.app.ui.playback.PlaybackPanel` reads CPU playback settings
and queues edits for the next frame boundary. The viewer uses Newton's native
GUI when no application panel is active. Picking and explicit frame capture
retain their on-demand readback.

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

The Newton adapters own the scene-object `render()`/`destroy()` protocol, shape
batching, layer-qualified names and GUI hooks. `newton_gl.py` also supplies the
current context, framebuffer size and resolved scene targets/camera data to the
fluid passes; those passes do not access Newton's private renderer fields.
Original particles draw in the opaque scene pass, sharing rigid
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

RSP adapts only its renderer instance to nonblocking event polling. The application
owns simulation and policy cadence; window/UI events continue
to be dispatched. Hidden COM markers skip Newton's otherwise unconditional
update/upload. Enabling the COM overlay still uses Newton's original drawing.

Newton owns model/layer cleanup and calls the object's `destroy()`. The adapter
also handles the window-close event before the context disappears. When updating
Newton, run the rendering tests to verify this internal scene-object contract.

## Verification and timing

Use the active WarpSim environment from the repository root:

```powershell
python tools/check_rsp_architecture.py
python -B -m unittest discover -s tests -p test_rsp_launcher_ui.py
python tools/run_rsp.py
```

The maintained CPU contracts check ownership and callback cleanup; they do not
validate CUDA/OpenGL interop or visual output. After renderer changes, check the
relevant mode, paused switches, reset and window close in the actual application.
The latest acceptance evidence is retained in
`output/architecture_completion_checks/REPORT.md` and `output/render_architecture.log`.
See [local-check policy](../README.md#running-and-checking) for the retained suite.

Generate targeted profiling probes only when needed, under ignored `output/`.
Compare identical scene/configuration and warmup conditions. Drain GPU work at
batch boundaries, distinguish CPU call time from GPU kernel timings, and keep
loading, JIT compilation and readback outside the measured interval. Do not run
recording tests or generate recording test artifacts.
