# RSP asset pipeline

Asset preparation, storage and read-only runtime loading have separate owners.
Geometry, SDF baking, volume integration and coupling retain their existing methods.

## Package installation and imports

The package lives in `RSP/sim_ready`, with the case-sensitive import name
`RSP.sim_ready`. From the repository root in the existing pinned WarpSim environment:

```powershell
conda run -n WarpSim python -m pip install -e . --no-deps --no-build-isolation
```

The `roboswimmingpool-rsp` distribution requires Python 3.11 or newer and uses
the six existing pins in `RSP/sim_ready/requirements.txt` as its dependency source.
Only RSP Python packages are included; MimicKit and asset data are not packaged.
This workflow supports an editable source checkout. Asset paths remain relative
to the repository, and standalone wheel data lookup is outside its scope.

```python
from RSP.sim_ready.paths import PBF_PLAY_ROOT
from RSP.sim_ready.lib.storage.store import AssetStore
from RSP.sim_ready.pipelines.boundary.prepare import prepare_boundary_asset
from RSP.sim_ready.pipelines.boundary.load import load_boundary_asset

store = AssetStore(PBF_PLAY_ROOT)
# Pass store=store to asset preparation; loading only needs an artifact reference.
```

The existing MimicKit launcher, backend and solver remain in place. Install the
RSP package before using the Newton-PBF backend or running the tests. The old
top-level `sim_ready` imports have been replaced; no compatibility package or
new command-line entry point is provided.

## Source and output layout

```text
data/assets_RSP/
├── .gitignore
├── origin/
│   ├── .gitignore
│   ├── bunny_flat_bottom.obj
│   └── spot_triangulated_good.obj
├── pbf_play/
│   ├── boundaries-v3/
│   ├── collision/
│   ├── primitives/
│   └── scene_manifest.json
└── tests/
```

The two OBJ files are the RSP source meshes. Both retain their filenames and
original SHA-256 values after relocation from `mimickit/sim_ready/Assets`.
The asset-root ignore rule excludes outputs; `origin` whitelists only the two
OBJs and its ignore rule. Code, hand-authored configs and recipes stay versioned
in their code/config directories. MimicKit models, motions, policies, paths and
ignore rules are unchanged. Training outputs, recordings and backend check
reports still use their existing locations.

The demo defaults to `data/assets_RSP/pbf_play`. Repository-local asset output
roots must be proper subdirectories of `assets_RSP`, excluding `origin` and its
descendants. The root itself is not an output directory. Explicit external
output roots are supported. Paths are resolved before checking containment.
The phase-two asset migration built this root from source and removed only the
obsolete `output/sim_ready/pbf_play/boundaries` directory after validation.
The subsequent move of Python code to `RSP.sim_ready` preserves these existing
assets and cache identities. Other legacy outputs remain outside the default
demo's asset path.

## Code layers

```text
RSP/sim_ready/
├── config.py                         shared parameter constraints
├── paths.py                          source locations and output rules
├── lib/
│   ├── asset_validation.py           plain-data checks, descriptions, diagnostics
│   ├── mesh/                         geometry, validation, convex algorithms
│   │   └── shapes.py                 collision shapes → body-local solid meshes
│   ├── volume/                       SDF baking and volume-map quadrature
│   └── storage/                      OBJ/NVDB/hull IO, fingerprints,
│                                     rollback publication, AssetStore
├── adapters/newton.py                public importer/finalize reuse and model extraction
└── pipelines/
    ├── articulation.py               all-link checks, coverage, boundary preparation
    ├── collision.py                  convex recipe, cache reuse and preparation
    ├── rigid.py                      primitive/rigid asset preparation
    ├── container.py                  terrain/container placement and preparation
    ├── boundary/
    │   ├── schema.py                 references, manifests and grid metadata
    │   ├── prepare.py                SDF selection, map construction, publication
    │   └── load.py                   read-only validation and GPU loading
    ├── scenes/pbf_play.py            active scene recipe and native geometry
    └── preset/                       historical internal references only
```

`lib.mesh.convex` provides `decompose_convex`, `validate_hulls` and
`measure_hull_error`. These operate in memory. `pipelines.collision.prepare_hulls`
owns the CoACD recipe and cache decisions, using storage's NPZ IO and publication.
Container-specific stairs, placement and wall recipes live in the container
pipeline; generic transforms, Boolean operations and geometry validation stay
in `lib.mesh`.

`AssetStore(output_root)` centralizes validated output paths, category directories,
recipe keys, temporary staging and group publication with rollback. It does not
decide whether an SDF covers a radius or whether collision data are valid.
Scene assembly creates a store and explicitly passes it to preparation calls.
Generic `rigid.build` and `container.build` take a relative `stem` and a `store`.
No compatibility modules are retained at the previous import paths.

## Active call sequence and interfaces

```text
PBFScene creates AssetStore
  → pipelines.scenes.pbf_play: primitive MJCFs and static geometry
    → pipelines.boundary.prepare.prepare_boundary_asset(..., store=store)
    → pipelines.collision.prepare_hulls(mesh, name, store)
  → Newton model assembly with the existing native collision shapes
    → adapters.newton: import + default finalize validation/correction + public DFS check
    → plain physical descriptions + separate runtime body-ID bindings
    → MimicKit skeleton/DOF compatibility checks before SimState and CUDA Graph creation
  → pipelines.articulation.prepare_articulated_asset
    → check every selected link before baking any link in that asset
    → lib.mesh.shapes → existing solid validation / Boolean union
    → pipelines.boundary.prepare with the same store and cache recipe
  → solvers.pbf.boundary_model.load_boundaries
    → pipelines.boundary.load.load_boundary_asset(artifact, runtime requirements)
  → body binding, independent instance transforms, pose sync and initialization
```

`lib.volume.pbf_volume_map.precompute_volume_map(sdf, support_radius,
quadrature_points=16)` returns an independent CUDA volume on the SDF grid.
Quadrature is keyword-only. The function performs no file IO and leaves the
input unchanged.

`prepare_boundary_asset(mesh, checked, *, store, ...)` accepts validated local
geometry, bake settings, container/solid treatment, support radius, quadrature,
device and provenance, and returns a `BoundaryArtifact` manifest reference.
`load_boundary_asset(artifact, ...)` validates and returns GPU grids, local bounds
and a visual mesh reference. It never writes or imports build algorithms, and
does not require the visual OBJ to exist. Runtime body IDs and placements belong
to the separate `scene_manifest.json`, not shared cache identities.

## Physical asset validation

`lib.asset_validation` only takes ordinary data/NumPy arrays. It has no dependency
on Newton, MimicKit, an environment or an agent. Its report distinguishes `pass`,
`warning`, `error`, `unsupported` and `not_checked`. Errors retain their underlying
exception and body/shape context. It checks explicit units/up-axis/scale, names,
finite values, dimensions, local transforms, coverage and consumer mappings.

`adapters.newton.import_physical_asset` uses the existing MJCF/URDF importer options.
`finalize_model` calls Newton's default `finalize()` and public
`validate_joint_ordering()`; it records warnings and inertia changes without
changing Newton's correction method. Source resource resolution and physical
structure validation remain with Newton. `describe_articulations(model)` copies
the actual `COLLIDE_SHAPES` geometry into local descriptions; runtime body IDs
are returned separately. Visual-only geometry is never substituted.
The current MimicKit backend requires one articulation per imported object and
one joint per body in matching order; incompatible valid Newton models are
reported as unsupported for this consumer before runtime state construction.

`prepare_articulated_asset(asset, *, store, bake_options, support_radius,
quadrature_points=16, device="cuda:0", excluded_links=())` returns dictionaries of
boundary references and local meshes keyed by asset-local body index, plus the
report. It accepts ordinary rigid bodies as one-link assets too. Sphere, box and
capsule tessellation/transform order is unchanged. Triangle meshes, convex meshes
and 32-section cylinders are supported. Mesh scale is applied once; nonuniform
and mirrored scales work, zero scale is rejected and winding follows reflection.
Imported solids reuse exact welding/validation and Manifold union; there is no
hole repair, reorientation or automatic proxy. Rounded barrel cylinders and
other unsupported geometry produce explicit diagnostics.

Missing native collision geometry fails by default. Auxiliary bodies must be
named in `excluded_links`; the demo exposes this as `pbf_excluded_links` in its
environment config. Exclusions remain visible in the run report. Each asset's
entire geometry is checked before its first boundary is baked. Static container
and bunny checks and rigid results are included in the scene report, published
only after successful loading/binding/sign checks. A failed preparation keeps
the previous successful scene manifest.

`PBFPlayEnv` supplies its already-loaded skeleton as `SkeletonDescription`.
Checks compare body counts/order/names, parents, joint types and DOF segments.
Floating-root coordinates and quaternion ball joints are handled separately
from MimicKit's non-root scalar/exp-map DOFs. Only selected controlled DOFs and
current control-mode restrictions are checked; passive/fixed joints do not need
motors. Motion data, policies and observations remain MimicKit's responsibility.
This is basic simulation-asset validation, not a self-intersection proof, automatic
rigging/physicalization or a guarantee that an existing policy fits a new robot.

The existing G1 MJCF passes structure/skeleton checks. Some G1 collision meshes
are open, nonmanifold or contain duplicate/collapsed triangles and are deliberately
rejected with link/shape names; valid limbs can be prepared independently.
No SimReady Foundation, Isaac Sim or new dependency is needed.

## Cache contracts

Within `boundaries-v3/<geometry-key>`, `sdfs/<key>` stores a single-grid SDF,
processed OBJ and `sdf.json`; `maps/<key>` stores a dual-grid NVDB and
`boundary.json`. Relative manifest references allow the artifact tree to move.
Preparation selects the smallest valid SDF support range covering both the
requested radius and `BakeOptions.max_support_radius`. Map keys include the
persisted SDF fingerprint, radius, quadrature and algorithm version.

Boundary loading checks file/content fingerprints, support metadata, transforms,
actual tile coordinates and local bounds. Preparation rebuilds invalid boundary
caches; loading reports an error. The completed manifest is published last.

Collision retains the existing CoACD options, version check, 20-character recipe
keys and `hulls.npz` / `metrics.json` format. Repeated preparation validates and
reuses hulls. Corrupt hull caches still raise; this phase adds no integrity
manifest, automatic collision repair or cache-format migration. Metrics are
published before hulls. Construction runs before simulation; concurrent writers
to the same artifact are outside the supported workflow.

## Verification

From the repository root in the WarpSim CUDA environment:

```powershell
conda run -n WarpSim python -m unittest discover -s tests -p test_rsp_asset_pipeline.py -v
conda run -n WarpSim python -m unittest discover -s tests -p test_pbf_boundary_assets.py -v
conda run -n WarpSim python -m unittest discover -s tests -p test_newton_pbf_refactor.py -v
conda run -n WarpSim python -m unittest discover -s tests -p test_rsp_asset_validation.py -v
conda run -n WarpSim python -m unittest discover -s tests -p test_articulated_asset_pipeline.py -v
```

Asset fixtures and their compilation caches live under `data/assets_RSP/tests`.
`tools/check_newton_pbf_refactor.py snapshot` and `compare` check demo state,
model mappings and reset without recording video. Backend reports retain their
existing output location. The optional `integration` command still checks policy
playback and recording, but is not part of routine asset/package migration checks.
Historical preset runtime restoration and application launch normalization remain
separate tasks.
