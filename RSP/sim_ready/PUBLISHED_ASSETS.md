# Published reference scene

## Same-window ImGui scene selector

From the activated conda environment, run `python tools/run_rsp.py`.
The default preset is the existing demo; presets bypass layout generation.
Custom assembly selects one published container and a humanoid/policy profile,
with enabled prop types each contributing 1–3 instances (unchecked means zero).
There is no additional total-count cap. Regular or seeded random layouts use
simple bounds in the existing water region; they avoid the static bunny and
other props without baking or exact collision queries. The humanoid retains
its existing GSI initialization. No file import or additional asset root is exposed.

The selection panel runs inside the Newton window. Start is handled after the
UI frame finishes: the panel is detached, the scene and policy are loaded, and
playback continues with the same Viewer, window and OpenGL context. Asset cards
use thumbnails generated from published display geometry in that context.
Reset resets the current scene. Closing the window exits. Initialization
failures clean up and report the exception in the console.

`data/playbacks/characters.yaml` selects the arguments for each profile.
`args/rsp_pbf_demo_args.txt` configures the default humanoid, and
`data/playbacks/presets.yaml` supplies preset labels. Policy weights are read from
the paths configured under `output`. Source assets stay in `origin`, complete
versioned assets in `published`, and preparation caches in `pbf_play`.
Presets are read in place. Custom scenes are saved to
`<demo out_dir>/scenes/<id>/scene.yaml`. No Env or
argument files are copied or rewritten. Old launcher outputs are left untouched.
The application owns the Viewer; Presentation borrows it. The original demo
entrypoint still creates and closes its own Viewer, with unchanged configuration.

Targeted launcher contracts:
`python -B -m unittest discover -s tests -p test_rsp_launcher_ui.py`.
Use `python tools/run_rsp.py` for interactive same-window checks. The one-off
acceptance scripts were retired; maintained checks are listed in
[local-check policy](../README.md#running-and-checking).

Run from the repository root in the existing WarpSim environment:

```powershell
python tools/prepare_rsp_assets.py
python mimickit/run.py --arg_file args/rsp_pbf_demo_args.txt
```

Preparation scans `data/assets_RSP/origin/**/source.yaml`. The reference recipes
cover the humanoid, G1, two rectangular pools with the suspended bunny, and three props.
They reuse the existing geometry, boundary and collision pipelines and the
`data/assets_RSP/pbf_play` cache. Original MimicKit assets remain untouched.

Complete packages are published under
`data/assets_RSP/published/<category>/<name>/<version>/asset.json`. Categories
are `articulations`, `props`, `containers`, and `environments`. The last category
is reserved; there is no new playground importer in this version. Packages
contain their physical models, visual/collision resources and selected boundary artifacts,
including their complete relative dependencies. Published versions are immutable: unchanged
recipes reuse verified packages; changed recipes require a new version.

`data/scenes/pbf_play.yaml` references explicit asset versions. The environment
configuration selects it through `scene_preset`; `pbf_asset_dir` now selects the
published root. Set `scene_preset` to one of these files before using the same launcher:

| Preset | Contents |
| --- | --- |
| `data/scenes/pbf_play.yaml` | Original 3x2x5 pool and three props |
| `data/scenes/pbf_empty.yaml` | Original pool and humanoid, no extra props |
| `data/scenes/pbf_multi.yaml` | Reordered props, a second rotated/moving light box, explicit observation ball |
| `data/scenes/pbf_wide.yaml` | 4x2x5 pool; original initial water region and props |

Instances have unique names and asset references, position, color, optional
XYZW unit quaternion `rotation`, `linear_velocity`, `angular_velocity`, and
`role: policy_sphere`. Orientation defaults to identity and velocities to zero.
Only one instance can carry the observation role. Enabling sphere observations
without that role is rejected at startup; runtime requests stay disabled with a
log message. Instance order does not define its role or resource binding.

The `fluid.start` and `fluid.end` fields use world Z-up coordinates. Containers
are static, meter-scale, origin-based rectangular cavities. Preflight validates
asset parameters, rotated prop bounds, prop overlap and spawn/fluid regions
before GPU state construction. Water and internal static geometry may initially
overlap props, preserving the original demonstration behavior. Models, density,
geometry and policy compatibility are not overridden by instance configuration.
The existing agent/checkpoint arguments and physical scalar defaults remain authoritative.

Catalog browsing reads metadata only. Scene assembly verifies each selected
package once. New publication fingerprints include source input hashes,
producer code under stable names, dependencies and the explicit support radius.
The verified older fingerprints for the seven current assets are accepted for
reuse when the corresponding new recipe is identical. Existing package
contents and version numbers are preserved. Old container
packages without assembly metadata must be explicitly replaced by a new version.

Playback resolves published resources, imports the physical models, validates
the skeleton and local physical descriptions, then binds published boundaries
to the new runtime body IDs. It never prepares missing assets. Run records go to
`output/pbf_play/runs/<session>/scene_manifest.json`, outside published packages.

Check the immutable publication contract:

```powershell
python -B -m unittest discover -s tests -p test_rsp_publication.py
```

For scene changes, select the relevant preset in `python tools/run_rsp.py`.
Existing packages are verified and reused by preparation; this is not a cold
bake. Use a separate publication/cache root when checking asset generation,
so existing versioned packages and saved scene references remain intact.

## G1 collision-based playback

Prepare only the new G1 package (existing packages are not republished):

```powershell
python tools/prepare_rsp_assets.py --asset articulations/g1/1
python tools/run_rsp.py --profile g1 --direct
```

The reference recipe reads `data/assets/g1/g1.xml` and its dependencies without
changing them. It publishes a self-contained model, authored visual meshes and
license under `published/articulations/g1/1`. Only the original collision
primitives generate coupling boundaries: 27 shapes on 21 bodies. The manifest
explicitly excludes the 10 collision-free bodies. All 31 bodies, their joints,
inertias and visuals remain in the simulation; no visual-mesh repair is used.

`data/playbacks/characters.yaml` associates a display name with an argument
file. The G1 profile uses the existing `output/g1_lafan_full_20261107_r01`
checkpoint and configuration, the current LaFAN prior, and the joint-limited
motions referenced by `data/datasets/dataset_g1_locomotion.yaml`. These files
remain at their original locations. Playback uses the shared SMPG1 agent state,
G1 LaFAN initialization and native 243-observation/29-action interface. It does
not run training or the original dodgeball projectile spawner.

The default engine runs 120 control frames and 480 rigid substeps per simulated
second. The 30 Hz policy supplies one action every four control frames. G1
keeps its saved position-target margins and knee limits. `pbf_g1.yaml` uses the
original pool, water region and prop arrangement. A reset retains GSI behavior;
it does not promise a fixed pose or standing stability.

Run `python tools/run_rsp.py` for selection in the existing Newton window.
Presets select their matching humanoid/policy automatically. Custom assembly
selects a complete humanoid profile, never a mesh with an unrelated policy.
Unavailable profile resources are reported without preventing the default
humanoid from running. Model and policy loading happen after selection.

Run the G1 profile with `python tools/run_rsp.py --profile g1 --direct`.
The prior one-off headless/GSI acceptance is recorded in
`output/architecture_completion_checks/REPORT.md`; it is not part of the small
CPU contract suite.
