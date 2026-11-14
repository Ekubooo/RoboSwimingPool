# Fluid playback clock

Edit `DEFAULT_SETTINGS` in `fluid_clock.py` before launching the application.
`FluidSettings.pbf` uses the existing WarpLab-compatible `PBFConfig`; physical
parameters have one definition. For example:

```python
DEFAULT_SETTINGS = FluidSettings(
    pbf=PBFConfig(particle_radius=0.01, substeps=2, pressure_iterations=3,
                  block_end=(1.5, 2.08, 1.25)),
    time_scale=0.2,
    feedback_scale=1.0,
    max_dt=1.0 / 300.0,
)
```

`time_scale` is relative to the unchanged MimicKit control clock. `frame_dt`
is derived from that multiplier; setting `PBFConfig.frame_dt` does not add a
second speed control. At control 30 Hz / rigid 120 Hz, defaults advance fluid
by 1/150 s per control frame in two 1/300 s steps, before rigid steps 2 and 4.
The humanoid advances by 1/30 s in four 1/120 s rigid steps. PD gains, target
updates at 30 Hz and torque limits are unchanged; feedback is evaluated with
each of the 120 Hz physics steps.
Higher speeds increase the actual step count
to respect `max_dt`; this bound alone does not guarantee numerical stability.

The water block spans height 0.50–2.08 m and contains 301,392 particles under
the existing lattice rule. `PBFConfig` itself retains the reference defaults;
only RSP's `DEFAULT_SETTINGS` selects this demo configuration.

All other PBF fields retain their names/defaults: density,
pressure regularization, gravity, viscosity, vorticity, maximum speed,
artificial pressure strength/reference distance, negative-pressure clamping,
container size, initial block bounds and velocity. Author vectors in WarpLab's
Y-up coordinates; the scene adapter converts them to the Z-up world. Fluid
gravity is independent of rigid gravity. Container dimensions must match the
existing pool assets. Radius changes use the existing asset preparation path;
this feature does not add arbitrary pool generation or alter that pipeline.

Physical coefficients are compilation constants in the existing solver.
Restart after changing configuration. This version has no parameter UI and
does not support hot reconfiguration of a solver or CUDA Graph.

## State and feedback

`ClockedPBFMuJoCo` inherits the existing composition adapter, replacing only
scheduling and graph bookkeeping. The engine's Newton State pair remains
authoritative. Skipped fluid updates copy particle state on GPU. Multiple fluid
steps share persistent scratch storage and end in the outgoing Newton State.

Each PBF update clears/replaces its force buffer. Rigid steps apply the latest
result once, multiplied by `feedback_scale`, to both force and torque. They do
not sum successive fluid forces or compensate for time scaling. Zero speed
skips fluid integration and feedback. Reset clears the fluid clock and forces.

Capture restores host counters on success or failure. Replay advances the
rigid counter by the configured rigid count and fluid counter by its own step
count. Odd fluid counts require two graphs for PBF prediction-buffer parity.
Odd rigid counts use the existing backend's final GPU copy to its persistent
views. The default four-step schedule does not require it. All graph parameters
are fixed at startup.

## Checks

Select checks relevant to the change; recording is not tested. Clock checks:
`python -m unittest discover -s tests -p test_rsp_runtime.py -v` in WarpSim.
The profiler accepts `--fluid-profile legacy|default|two|paused|fast` for
separate-process checks. `legacy` uses the original eight 1/240 s fluid steps.
`--eager` disables graph replay. `--stability-frames 1800` checks 60 main-clock
seconds and saves checkpoints outside timed samples; `--reset-at` exercises
reset during that check.

## Continuous policy playback

`playback.PlaybackRunner` is shared by the PBF demo entry point and the playback
profiler. It calls the playback policy adapter and environment, without episode-return
tracking, GPU completion-condition queries or automatic resets. `PBFPlayEnv`
retains actor observations and GSI reset data, but does not update or publish
training histories/features. Other MimicKit environments keep their test loop.

`ObservationBuilder` borrows the backend's persistent state views and replays
captured Torch operations into the actor observation buffer. Joint mapping and
buffers are created once. `observation_math` preserves MimicKit's scripted math,
replacing scalar assignments with GPU `fill_` operations so that fixed axes do
not trigger host transfers/waits. The two projectile modes have separate graphs.
`PolicyInference` borrows the loaded actor mean network and both
original normalizers; action clipping remains in the environment. No distribution,
log-probability or training metadata is built on the hot path. Model loading and
GSI initialization/reset still use the original MimicKit implementation.

The runner places Warp work on the Torch execution stream. Scoped GPU events
preserve dependencies with initialization/reset work on other Warp streams;
normal frames do not require a CPU synchronization for observation construction.
The original `_compute_obs` remains available for construction and numerical
comparison, but ordinary playback updates use the GPU builder.

The viewer owns pause/single-step state. Enter toggles pause, Space advances
one control frame and stays paused, and R queues a reset. Reset and policy-input
switches take effect between frames, before policy inference. Paused frames
still draw and process camera/UI events. The application's 30 Hz limiter is
unchanged; only performance tools bypass it for uncapped measurements.

Use `tools/profile_rsp_playback.py --label NAME --loop playback --fluid default`
for the real playback loop, and `--fluid old` to isolate code changes using the
reference 195,525 particles / three substeps. `--loop legacy` is for an actual
pre-change checkout. `--separate-wait` is an attribution diagnostic, not a
throughput benchmark. All outputs are under `output/rsp_render_profile/NAME`.

`--rigid-substeps 8` selects the previous physics frequency for an isolated
comparison without editing production configuration. Each profiler run measures
paced and uncapped playback; `--modes uncapped` selects only that mode. Screenshots
are opt-in with `--screenshot`. The targeted inference check is
`python -m unittest discover -s tests -p test_rsp_inference.py -v`; the real-scene
check `python tools/check_rsp_playback.py` covers policy equivalence, persistent
state views and one 900-frame (30 main-second) trajectory, without recording.
See [the stage 5 results](../../docs/rendering_stage5_results.md) for the measured
comparison and the reason four rigid steps replaced the unstable three-step trial.
