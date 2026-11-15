# Fluid playback clock

Edit `DEFAULT_SETTINGS` in `fluid_clock.py` before launching the application.
`FluidSettings.pbf` uses the existing WarpLab-compatible `PBFConfig`; physical
parameters have one definition. For example:

```python
DEFAULT_SETTINGS = FluidSettings(
    pbf=PBFConfig(particle_radius=0.01, substeps=2, pressure_iterations=3,
                  block_start=(1.56, .50, .10), block_end=(2.96, 2.08, 1.25)),
    time_scale=1.0,
    feedback_scale=1.0,
    max_dt=1.0 / 300.0,
)
```

`time_scale` is relative to the simulation frame clock. `frame_dt` is derived
from that multiplier; setting `PBFConfig.frame_dt` does not add a second speed
control. In `data/engines/newton_pbf_engine.yaml`, the backend's legacy
`control_freq: 150` names the frame frequency, not the policy frequency;
`sim_freq: 600` selects four rigid steps of 1/600 s per frame. Both rigid and
fluid time advance by 1/150 s each frame. Fluid runs in two 1/300 s steps,
before rigid steps 2 and 4. PD gains and torque limits are unchanged; PD runs
each rigid step, at 600 Hz in simulated time.

`playback.POLICY_HZ` remains 30. The runner derives five frames per decision
from the engine timestep, holds the GPU action between decisions, and retains
the environment's action clipping. Pause preserves the decision phase; reset
discards the held action and makes the next advancing frame infer immediately.
Policy-input switches affect the next scheduled decision.

`SimEnv.get_policy_timestep()` normally returns the engine frame timestep;
`PBFPlayEnv` overrides it to 1/30 s. SMP's prior compatibility check, GSI
initial velocity reconstruction and reference history sampling use that policy
period, preserving the existing 30 Hz data even with 150 Hz simulation frames.

Playback is uncapped: Newton-PBF renders without invoking the base Engine's
sleep. It advances a fixed amount per displayed simulation frame, without a
wall-clock accumulator. At 100 FPS it advances 2/3 simulated second per real
second; 150 FPS is real time. Slow motion therefore depends on actual FPS.
Higher speeds increase the actual step count
to respect `max_dt`; this bound alone does not guarantee numerical stability.

The right-side water block spans world bounds X=1.56–2.96, Y=0.75–1.90,
Z=0.50–2.08 m. The existing 0.02 m lattice yields 301,392 particles. The initial block deliberately
includes the static bunny's interior, without filtering or changing the solver.
`PBFConfig` itself retains the reference defaults; only RSP's `DEFAULT_SETTINGS`
selects this demo configuration. `play_spawn_xy` in the environment YAML places
the humanoid at (0.65, 1.0) on the left, preserving MimicKit's sampled root
height, pose and velocities instead of lifting it above the water.

All other PBF fields retain their names/defaults: density,
pressure regularization, gravity, viscosity, vorticity, maximum speed,
artificial pressure strength/reference distance, negative-pressure clamping,
container size, initial block bounds and velocity. Author vectors in WarpLab's
Y-up coordinates; the scene adapter converts them to the Z-up world. Fluid
gravity is independent of rigid gravity. Container dimensions must match the
existing pool assets. Radius changes use the existing asset preparation path;
this feature does not add arbitrary pool generation or alter that pipeline.

Other physical coefficients remain compilation constants in the existing solver;
restart after editing them. Runtime controls below only edit fluid gravity,
feedback scaling and the fluid timestep, without changing the numerical solver.

## Interactive fluid controls

The panel keeps Pause, Step, Reset and Policy: scene sphere. Gravity strength
ranges from 0 to 20 m/s² (default 9.81); release the slider to apply it without
changing direction. Feedback ranges from 0 to 2 (default 1), with live updates
to a persistent GPU scalar. Zero disables feedback but leaves rigid-to-fluid
interaction enabled. No GPU state is read back to populate the panel.

G reverses fluid gravity. O/P select world -X/+X; press the same direction key
again to restore -Z. G clears the horizontal selection. X and the slow-motion
checkbox toggle fluid-only half speed: two 1/600 s fluid steps per frame,
instead of two 1/300 s steps. Rigid steps and the five-frame policy interval
stay unchanged. Feedback uses the actual fluid dt, then the separate multiplier;
no speed compensation is applied. The fluid clock accumulates elapsed time
across speed changes without reinterpreting past ticks.

R and the Reset button share the same queued reset: restore default gravity,
feedback and speed, reset both clocks and the scene, clear single-step requests,
and pause. Scene-sphere selection is preserved. Initial application startup
still runs normally. UI text input suppresses application shortcuts; held
toggle keys do not repeat, and releasing a key or losing focus clears it.

`FluidInteraction` owns CPU settings/commands. `PlaybackRunner` binds it to the
existing solver and UI, and applies edits on the playback stream between frames.
Gravity and speed edits invalidate `ClockedStepGraph`, which recaptures lazily
on a Warp-owned stream before advancing again. The capture preserves counters,
elapsed time and prediction-buffer parity. A failed capture pauses playback
and reports an error; it does not launch the stale graph. Feedback-only edits
update device data and never recapture. Human/model gravity is not modified.

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
views. The default four-step schedule does not require it. Interactive edits
leave the number of fluid and rigid calls unchanged.

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
one 1/150 s simulation frame and stays paused, and R resets defaults and pauses. Reset and
policy-input switches take effect between frames. Paused frames still draw and
process camera/UI events. The application's former 30 Hz limiter is removed;
other engine backends retain their existing behavior.

Use `tools/profile_rsp_playback.py --label NAME --loop playback --fluid default`
for the real playback loop. `--fluid old` selects reference `FluidSettings`
(195,525 particles, minimum three substeps, time scale 0.2), with the current
frame clock; it does not restore the old complete configuration. `--loop legacy` is for an actual
pre-change checkout. `--separate-wait` is an attribution diagnostic, not a
throughput benchmark. All outputs are under `output/rsp_render_profile/NAME`.

`--rigid-substeps 8` overrides only the rigid step count at the configured frame
frequency, without editing production configuration. The profiler measures the
uncapped production loop directly, without overriding its pacing. Screenshots
are opt-in with `--screenshot`. The targeted inference check is
`python -m unittest discover -s tests -p test_rsp_inference.py -v`; the real-scene
check `python tools/check_rsp_playback.py` covers policy equivalence, persistent
state views, five-frame action holds and one 1500-frame (10 simulated-second)
trajectory, without recording.
See [the stage 5 results](../../docs/rendering_stage5_results.md) for the measured
comparison and the reason four rigid steps replaced the unstable three-step trial.
The [stage 6 results](../../docs/rendering_stage6_results.md) cover uncapped
playback, matched 1/150 s frame clocks and five-frame policy decisions.
