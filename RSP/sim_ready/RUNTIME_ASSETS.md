# Runtime assets for the reference scenes

Use the activated WarpSim environment from the repository root:

```powershell
python tools/run_rsp.py
python tools/run_rsp.py --profile g1 --direct
```

Startup scans `origin/props` recursively for OBJ files, prepares their validated
local meshes, collision models and SDFs, then opens scene selection. Invalid
files are listed and excluded. Choose a preset or assemble a scene from a container,
humanoid profile and props. Regular and seeded random layouts use source
bounds and avoid static terrain/models and other props. The application prepares
volume maps only for the selected assets and continues into simulation in the same window.

Preparation is automatic. For an explicit preparation step, use
`python tools/prepare_rsp_assets.py --asset props/scene_sphere` or omit
`--asset` to prepare every declared source. The CLI uses the same support radius
and generated directory as playback.

Use `python tools/prepare_rsp_assets.py --sdf-only` to perform just OBJ
preprocessing, without computing volume maps.

## Inputs and layout

Place new Y-up OBJ files under `data/assets_RSP/origin/props`; no per-OBJ YAML is
required. Filenames must be unique (case-insensitive), including built-in prop
names. Dynamic OBJ props are uniformly scaled to a 0.3 m longest edge, centered
and converted to Z-up. They use the existing prop-color rendering path; this
import path does not add material/texture authoring.

Articulation declarations retain their category, name, recipe and file inputs.
A source may refer to other
approved repository assets, such as the authored G1 model under
`data/assets/g1`. Simulations still read the configured policies and motions.

Scene YAML refers to sources by category and name. Older scene files with a
valid `version` field select the current asset of that category and name.
New custom scenes are saved under the selected profile's output directory.
The stored version field no longer selects historical geometry.

| Preset | Scene |
| --- | --- |
| `pbf_play.yaml` | Reference pool, humanoid and three props |
| `pbf_empty.yaml` | Reference pool and humanoid |
| `pbf_multi.yaml` | Reordered props, a rotated moving box and a policy sphere |
| `pbf_wide.yaml` | Wider pool, with the original water region |
| `pbf_g1.yaml` | G1 with its own policy and original props |

Instances store their position, color, optional XYZW quaternion and linear and
angular velocities. A `policy_sphere` role explicitly binds a single prop to the
sphere observation. The scene preflight checks the spawn, fluid region, rotated
prop bounds and prop overlap. Water and internal static geometry may initially
overlap authored props.

## Terrain containers and generation

Select `pbf_stairs_full` for continuous full-floor stairs, with a central 0.7 m
platform. Default level tops are 0.1, 0.2 and 0.3 m, in the same 3 x 2 x 5 m
cavity with 0.3 m walls. The existing local `pbf_stairs` option remains available.

In **Custom assembly**, choose **Generate terrain** to open an ImGui modal in
the current window. Stairs exposes step height; Random grid exposes maximum
height and seed; Gap exposes gap width and depth. The preview uses the same
orthographic camera as the container cards. Confirm **Save terrain** with a
new name to save `origin/terrains/<name>/terrain.yaml` and select its card.
Cancel/Escape leaves no source file. Reopening the app discovers saved recipes.
The source recipe is retained; meshes, collision geometry and dual-grid NVDBs
are generated in `runtime/containers/<name>` when simulation starts.

Random grid uses closed 0.5 m cells, 6 x 4 across the floor. An optional OBJ is
always placed in the first cell, centered at (0.25, 0.25), with its horizontal
extent fitted within 95% of the cell. Stairs and Gap use the central platform
instead. Gap raises its border and island above the fixed floor so the trench
remains open. Terrain and optional OBJ use a Boolean union; rigid collisions
combine native terrain boxes and reused OBJ convex pieces under the same
transform, without decomposing the final union again. This reuse does not
apply to subtractive Boolean operations, which are outside this editor.

### Legacy local stairs

In custom assembly select `pbf_stairs`, then optionally choose a discovered OBJ
as the central platform model. The 3 x 2 x 5 m pool has 0.3 m walls. Its Newton
stair patch is 1.6 x 1.6 m, centered at (2.05, 1.0), with 0.15 m step widths,
0.06 m heights and a 0.7 m platform. The optional model has a 0.665 m maximum
horizontal span, is centered on the platform and sinks 0.003 m into it.

Scene YAML stores the optional `terrain_prop: {category: props, name: ...}`
reference. It is a container dependency, not an extra dynamic object. The
generated `landscape.obj` excludes walls and the lid; the boundary mesh includes
the closed enclosure. Changing the central model replaces this container's one
current runtime slot. A hand-authored scene must match the prepared selection.

Legacy stairs preserve the character's flat spawn area. New full-floor terrains
lift the sampled character above overlapping obstacles at reset, preserving
its horizontal spawn, articulated pose and velocity. If necessary, assembly
raises the entire water block above the terrain/model, preserving its size and
particle count, before placing props. An arrangement that exceeds the ceiling
is rejected. Loading a hand-authored scene validates placement without rewriting
it. Existing reference presets keep their original layout rules.

## Generated data and invalidation

The default `pbf_asset_dir` is `data/assets_RSP/runtime`. Each source retains
one current set of generated files. An unchanged build is verified and reused.
Changes to inputs, build code, dependencies or support radius generate a new
current set when relevant to that asset. Terrain-specific code changes do not
invalidate unrelated humanoids or primitives. SDF, volume-map and convex-piece
identities allow verified stages to be reused independently. Caches made before
stage fingerprints were introduced require a one-time rebuild on first use.
Each preparation call uses its starting source declarations and
build code. A terrain recipe changed after discovery is rejected until it is
discovered again. Preparation never edits the original model, meshes, motions or policy
files. Old `published` and `pbf_play` asset directories are outside the
active workflow. Simulation reports remain under `output/pbf_play/runs`.

The G1 source uses its native collision primitives for the coupled boundaries.
The runtime metadata records 21 participating bodies and 10 explicit
collision-free exclusions. All 31 bodies remain in the Newton articulation
with the original inertias, joints and visuals.

The default engine uses 150 control frames and 600 rigid substeps per
simulated second. The 30 Hz policy supplies one action every five control
frames. Asset reuse is independent of policy cadence.
