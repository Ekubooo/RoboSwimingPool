# Newton simulation and policy timing

Scene files define the published asset references, placement and initial fluid
region. The simulation resolves these references before it builds the Newton
model. It keeps a permanent primary state and a second state for substeps.
With an odd substep count, the final state is copied back to the primary arrays
on the GPU so borrowed state views keep their addresses.

`RSP.physics.fluid_clock.DEFAULT_SETTINGS` holds the fluid defaults. The
application passes those settings to the simulation and asset preparation.
The reference engine configuration uses 120 simulation frames per simulated
second and four 1/480-second rigid steps per frame. The fluid clock normally
runs two 1/240-second substeps, before rigid steps two and four. The reference
scene defines a particle lattice of 498,456 particles.

The 30 Hz policy decides every four simulation frames and holds its GPU action
between decisions. The MimicKit environment retains action clipping, actor and
normalizer loading, observation order, and GSI. GSI uses the policy sampling
interval when it reconstructs velocities and samples motion history. Advance
and render remain part of the normal environment step. Paused playback renders
without advancing either clock. Playback has no wall-clock pacing.

## Fluid controls and reset

The UI queues changes to gravity, fluid feedback and slow motion. The playback
loop applies them between frames on its Torch/Warp stream. Slow motion halves
the fluid timestep while preserving the substep counts, the rigid clock and
the policy cadence. Feedback applies the most recently computed fluid wrench
once per rigid step, multiplied by a persistent GPU scalar.

Gravity and slow motion change the captured work. The next advancing frame
refreshes the CUDA Graph before it evaluates the policy. If capture fails,
playback pauses and reports the error. Feedback-only updates leave the graph
valid. Solver buffers, captured kernels and prediction-buffer parity remain
under the physics module.

Reset restores fluid settings and clears held actions, clocks and feedback.
MimicKit first samples the reset pose and velocity, then applies the scene's
horizontal spawn position and resets solver histories. The simulation
synchronizes the secondary state with the primary afterwards.

## Backend interfaces

`SceneBuilder` records object-to-body IDs after the model has been validated.
The boundary binder and presentation use the same mapping. The Newton runtime
accepts a control object and `StepHooks` for adapter operations at the existing
frame and substep boundaries. Its contact sensor and observables provide raw
feedback arrays. The MimicKit adapter creates and owns the persistent object
views, command buffers and joint representation conversions.

See [RSP architecture](../README.md) for the launch and checks. Historical
profile results document the settings of those earlier versions.
