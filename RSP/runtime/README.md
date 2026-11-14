# Fluid playback clock

Edit `DEFAULT_SETTINGS` in `fluid_clock.py` before launching the application.
`FluidSettings.pbf` uses the existing WarpLab-compatible `PBFConfig`; physical
parameters have one definition. For example:

```python
DEFAULT_SETTINGS = FluidSettings(
    pbf=PBFConfig(substeps=3, pressure_iterations=3, viscosity=0.5),
    time_scale=0.2,
    feedback_scale=1.0,
    max_dt=1.0 / 300.0,
)
```

`time_scale` is relative to the unchanged MimicKit control clock. `frame_dt`
is derived from that multiplier; setting `PBFConfig.frame_dt` does not add a
second speed control. At control 30 Hz / rigid 240 Hz, defaults advance fluid
by 1/150 s per control frame in three 1/450 s steps, before rigid steps 3, 6, 8.
Two base substeps use 1/300 s each. Higher speeds increase the actual step count
to respect `max_dt`; this bound alone does not guarantee numerical stability.

All other PBF fields retain their names/defaults: particle radius, density,
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
rigid counter by eight and fluid counter by its own step count. Odd fluid
counts require two graphs for PBF prediction-buffer parity even with an even
rigid count. All graph parameters are fixed at startup.

## Checks

Run `python -m unittest discover -s tests -p test_rsp_runtime.py -v` in WarpSim.
The profiler accepts `--fluid-profile legacy|default|two|paused|fast` for
separate-process checks. `legacy` uses the original eight 1/240 s fluid steps.
`--eager` disables graph replay. `--stability-frames 1800` checks 60 main-clock
seconds and saves checkpoints outside timed samples; `--reset-at` exercises
reset during that check.
