# RoboSwimmingPool architecture

RSP runs existing MimicKit policies in a single Newton fluid/rigid scene. The
package is maintained in this repository and uses the existing pinned WarpSim
environment. Training, multiple simulation worlds and standalone wheel asset
discovery are outside this application's scope.

## Code ownership

| Package | Responsibility |
| --- | --- |
| `app` | Launch, CPU UI commands, playback scheduling and application lifetime |
| `simulation` | Scene description/resolution, Newton model/state and coupled execution |
| `integrations.mimickit` | Environment/agent loading, GSI, policy observations/actions and Engine compatibility |
| `physics` | Existing PBF, runtime boundary queries, coupling and fluid clock |
| `rendering` | CUDA/OpenGL resources, render passes, Newton viewer/GUI adapters and asset previews |
| `sim_ready` | Geometry/volume algorithms, asset preparation/publication and read-only loading |

The application composes these packages. Asset processing does not import the
application, solver or MimicKit. Rendering and simulation do not import
MimicKit. Physics may use the asset package's read-only boundary format and
loader. Provider-specific representation conversions remain in the integration.
Public imports have one canonical implementation; legacy Engine/environment
entry points only forward to it.

## Configuration and data flow

The application reads the existing playback profiles. Fluid defaults remain in
`physics.fluid_clock.DEFAULT_SETTINGS`; the application reads the current value
and passes settings explicitly. It derives the boundary support radius with the
existing PBF configuration code and supplies that value to asset preparation and
scene validation. The asset package never discovers a running solver.

The bundled playback presets use a 0.0125 m particle radius and a 55 x 45 x 141
lattice (348,975 particles). Their Z-up fluid bounds are (1.56, 0.75, 0.50) to
(2.96, 1.90, 4.05). Each main frame advances both systems by 1/150 s: four rigid
substeps at 1/600 s and two fluid substeps at 1/300 s, with three pressure
iterations per fluid substep. The policy stays at 30 Hz (every five frames).
These settings require volume maps baked with support radius 0.05 m; the
bundled source manifests and scene references select new immutable versions.

Volume-map precomputation belongs to `sim_ready`: its numerical function works
in memory, the preparation pipeline writes SDF plus volume-map NVDB resources,
and the boundary loader validates and reads them. Runtime body transforms,
queries and fluid/rigid forces belong to physics. Published assets remain
immutable under `data/assets_RSP/published`; source meshes and preparation
caches retain their existing locations. Loading a scene never bakes assets.

Automatic placement avoids the internal bunny and uses its existing clearance.
Authored scene validation retains its separate exterior/prop-overlap rules.
These paths share geometry calculations without changing placement policy.

## Runtime and resource lifetime

The application owns the selection window and lends it to the environment.
Selection finishes its frame and releases preview textures before simulation
construction. The legacy MimicKit launch path still lets the environment own
its viewer. A playback binding only installs/removes application callbacks.

Newton state pairs and contact feedback have a single simulation owner.
MimicKit borrows persistent Torch views and owns its representation conversion
buffers. State storage, CUDA Graph inputs and observation buffers retain their
addresses across steps and resets. UI only changes CPU state; commands are
applied at a complete frame boundary on the playback stream.

The playback loop holds actions between policy decisions. The environment's
existing `step` also renders; the application does not draw a second successful
frame. Paused frames and failed graph updates use the explicit render-only path.
Rendering keeps device-side packing and CUDA/OpenGL mapping without CPU particle
readback. CUDA registrations and GL resources are released before their context.

## Running and checking

From the repository root, using the existing WarpSim environment:

```powershell
python -m pip install -e . --no-deps --no-build-isolation
python tools/run_rsp.py
python tools/run_rsp.py --profile g1 --direct
python tools/prepare_rsp_assets.py
```

The distribution includes RSP code, not MimicKit or asset data. Asset dependency
pins remain in `sim_ready/requirements.txt`. Playback uses the existing MimicKit
environment and Torch dependencies; rendering uses Newton's installed OpenGL
and ImGui dependencies. This refactor does not change any dependency version.

Run only checks relevant to a change. The architecture checker parses imports
without initializing CUDA or GL. Its optional `--solver-baseline REV` verifies
that a Solver relocation changed only the explicitly allowed import/module
paths. Tests and standalone checking/profiling scripts are local-only and ignored
by Git; a fresh checkout does not include them. Existing local copies may be
restored from a workspace backup, or regenerated for the change under review.
Commands in module guides that reference these scripts require those local copies.

The local contracts cover immutable publication, playback cadence/reset, Viewer
ownership, fluid-clock/Graph bookkeeping, and stable state storage. One-off probes,
traces and timing reports belong under ignored `output/`. Shared evaluation
libraries stay versioned: prior scoring lives in `tools/getup/prior_scoring.py`,
while the G1 evaluator also uses its control-probe and joint-diagnostic modules.
Recording interfaces are retained; recording tests and artifacts are excluded.
