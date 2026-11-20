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
| `sim_ready` | Geometry/volume algorithms, automatic asset preparation and read-only loading |

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

The bundled playback presets use a 0.01 m particle radius and a 69 x 56 x 91
lattice (351,624 particles). Their Z-up fluid bounds are (1.56, 0.75, 0.50) to
(2.96, 1.90, 2.34). Each main frame advances both systems by 1/150 s: four rigid
substeps at 1/600 s and two fluid substeps at 1/300 s, with three pressure
iterations per fluid substep. The policy stays at 30 Hz (every five frames).
These settings require volume maps baked with support radius 0.04 m. Before
creating an environment, playback prepares the selected scene's assets using
the same fluid settings as the simulation. Matching inputs, recipes and files
are reused. Missing, damaged or outdated outputs are regenerated; changing the
particle radius automatically updates the volume map. Before scene selection,
startup discovers OBJ props and prepares their validated meshes, collisions and
SDFs. Failures are shown and excluded. Volume maps are completed only after
assembly; previews use prepared OBJ geometry and original humanoid sources.

Volume-map precomputation belongs to `sim_ready`: its numerical function works
in memory, the preparation pipeline writes SDF plus volume-map NVDB resources,
and the boundary loader validates and reads them. Runtime body transforms,
queries and fluid/rigid forces belong to physics. Each asset has one current
directory, `data/assets_RSP/runtime/<category>/<name>`, with internal build
metadata and checksums. Prepared OBJ boundaries initially have one SDF grid;
simulation requires the completed SDF and volume map as two grids in one NVDB
file. An adequate existing SDF can be reused when only the volume
map needs updating. No asset version selection or publication step is required.

Original inputs stay outside the generated tree. Runtime directories can be
removed while the application is closed and rebuilt from OBJ files under
`data/assets_RSP/origin/props`, humanoid declarations, referenced originals and
the built-in generation recipes. Preparation
validates a temporary replacement before installing it and holds a directory
lease through environment use. Scene assembly and simulation remain read-only.
Legacy scene references with a version field select the current asset without
rewriting custom scene files.

Automatic placement avoids the internal bunny and uses its existing clearance.
Authored scene validation retains its separate exterior/prop-overlap rules.
These paths share geometry calculations without changing placement policy.
The additional fixed `pbf_stairs` container uses Newton-generated steps and an
optional source OBJ on its platform. Its custom layout preserves the flat spawn
area and raises the initial water block when needed without changing its size.
The saved scene records that placement and its optional `terrain_prop` reference.

`simulation/scenes/assembly.py` owns selection-time layouts; the UI only gathers
choices and displays results. Shared terrain geometry lives in
`sim_ready/container_geometry.py`, while `pipelines/container.py` validates and
persists generated assets. Preview and assembly do not import baking pipelines.
Newton terrain mesh conventions are isolated in `sim_ready/adapters/terrain.py`.
The full-floor stairs, random-grid and gap recipes use the same geometry path.

Asset fingerprints are scoped to the producing recipe and numerical stage.
SDFs, volume maps and convex pieces are independently checked before reuse.
Existing caches without these stage fingerprints are rebuilt on first use;
each asset still has just one current runtime directory.

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
An invalid sampled terrain placement leaves playback paused until a successful
reset; initial placement errors return to scene selection.
Rendering keeps device-side packing and CUDA/OpenGL mapping without CPU particle
readback. CUDA registrations and GL resources are released before their context.

## Running and checking

From the repository root, using the existing WarpSim environment:

```powershell
python -m pip install -e . --no-deps --no-build-isolation
python tools/run_rsp.py
python tools/run_rsp.py --profile humanoid --direct
python tools/run_rsp.py --profile g1 --direct
# Optional: prepare one asset without starting playback.
python tools/prepare_rsp_assets.py --asset props/scene_sphere
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

The local contracts cover automatic preparation and replacement, playback cadence/reset, Viewer
ownership, fluid-clock/Graph bookkeeping, and stable state storage. One-off probes,
traces and timing reports belong under ignored `output/`. Shared evaluation
libraries stay versioned: prior scoring lives in `tools/getup/prior_scoring.py`,
while the G1 evaluator also uses its control-probe and joint-diagnostic modules.
Recording interfaces are retained; recording tests and artifacts are excluded.
