# Runtime assets for the reference scenes

Use the activated WarpSim environment from the repository root:

```powershell
python tools/run_rsp.py
python tools/run_rsp.py --profile g1 --direct
```

The scene selector reads the source catalog and draws thumbnails before
runtime assets exist. Choose a preset or assemble a scene from a container,
humanoid profile and props. Regular and seeded random layouts use source
bounds and avoid the static bunny and other props. The application prepares
only the selected assets and continues into simulation in the same window.

Preparation is automatic. For an explicit preparation step, use
`python tools/prepare_rsp_assets.py --asset props/scene_sphere` or omit
`--asset` to prepare every declared source. The CLI uses the same support radius
and generated directory as playback.

## Inputs and layout

Source declarations live under `data/assets_RSP/origin`. They give the
`category`, `name`, recipe and file inputs. A source may refer to other
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

## Generated data and invalidation

The default `pbf_asset_dir` is `data/assets_RSP/runtime`. Each source retains
one current set of generated files. An unchanged build is verified and reused.
Changes to inputs, build code, dependencies or support radius generate a new
current set. Each preparation call uses its starting source declarations and
build code. Source declaration changes made while it runs apply on the next
call. Preparation never edits the original model, meshes, motions or policy
files. Old `published` and `pbf_play` asset directories are outside the
active workflow. Simulation reports remain under `output/pbf_play/runs`.

The G1 source uses its native collision primitives for the coupled boundaries.
The runtime metadata records 21 participating bodies and 10 explicit
collision-free exclusions. All 31 bodies remain in the Newton articulation
with the original inertias, joints and visuals.

The default engine uses 150 control frames and 600 rigid substeps per
simulated second. The 30 Hz policy supplies one action every five control
frames. Asset reuse is independent of policy cadence.
