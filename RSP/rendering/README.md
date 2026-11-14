# RSP particle rendering

`BillboardBatch` owns one CUDA/OpenGL instance buffer, a quad and its shaders.
`NewtonBillboardViewer` adapts that batch to Newton 1.7's existing scene, window,
camera and UI. MimicKit's `PBFViewer` name remains the application entry point;
the same viewer is used by the headless video recorder.
Importing the backend does not initialize OpenGL; a context is needed only when
constructing a viewer, so non-visual simulation retains its display-free path.

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
`render()`/`destroy()` protocol, camera matrices, layer-qualified particle names,
and `_log_particles()`. Particles draw in the opaque scene pass, sharing rigid
depth before transparency, MSAA resolve and ImGui. Human geometry, scene lines,
input callbacks and recording retain their Newton paths. No global monkeypatch
or modified installed package is needed.

Newton owns model/layer cleanup and calls the object's `destroy()`. The adapter
also handles the window-close event before the context disappears. When updating
Newton, run the rendering tests to verify this internal scene-object contract.

## Verification and timing

Use the project's WarpSim environment from the repository root:

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
