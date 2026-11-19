# RSP asset pipeline

The playback entry now uses versioned published assets. See
[Published reference scene](PUBLISHED_ASSETS.md) for preparation, preset selection,
read-only runtime loading and the targeted checks. The preparation/cache details
below remain applicable; runtime no longer invokes their build entry points.

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
from RSP.sim_ready.pipelines.boundary import prepare_boundary_asset
from RSP.sim_ready.boundary.load import load_boundary_asset

store = AssetStore(PBF_PLAY_ROOT)
# Pass store=store to preparation, and an artifact plus support radius to loading.
```

Install the RSP package before using the Newton-PBF backend or running the
tests. Use `tools/prepare_rsp_assets.py` to prepare published assets and
`tools/run_rsp.py` for playback.

## Source and output layout

```text
data/assets_RSP/
├── .gitignore
├── origin/
│   ├── .gitignore
│   ├── bunny_flat_bottom.obj
│   ├── spot_triangulated_good.obj
│   └── <category>/<name>/source.yaml
├── pbf_play/
│   ├── boundaries-v3/
│   ├── collision/
│   └── primitives/
├── published/
│   └── <category>/<name>/<version>/asset.json
└── tests/
```

The two OBJ files are the RSP source meshes. Both retain their filenames and
original SHA-256 values after relocation from `mimickit/sim_ready/Assets`.
The asset-root ignore rule excludes generated outputs. Source manifests, the
humanoid model referenced by its manifest, and the two OBJ files are under
`origin`. The G1 recipe resolves its input model separately from `data/assets/g1`.

Preparation caches intermediate data in `data/assets_RSP/pbf_play`. Playback
reads versioned packages in `data/assets_RSP/published`. Repository-local asset output
roots must be proper subdirectories of `assets_RSP`, excluding `origin` and its
descendants. The root itself is not an output directory. Explicit external
output roots are supported. Paths are resolved before checking containment.
Playback diagnostics live under `output/pbf_play/runs`.

## Code layers

```text
RSP/sim_ready/
├── config.py                         shared parameter constraints
├── catalog.py                        asset discovery and package verification
├── paths.py                          source locations and output rules
├── reference.py                      shared reference pool coordinates
├── boundary/
│   ├── schema.py                     references, manifests and grid metadata
│   └── load.py                       read-only validation and GPU loading
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
    ├── boundary.py                   SDF selection, map construction and cache
    ├── publication.py                immutable package publication
    └── recipes/                      reference container, prop and G1 production
```

`lib.mesh.convex` provides `decompose_convex`, `validate_hulls` and
`measure_hull_error`. These operate in memory. `pipelines.collision.prepare_hulls`
owns the CoACD recipe and cache decisions, using storage's NPZ IO and publication.
Container-specific stairs and asset geometry live in the container pipeline
and recipes. Generic transforms, Boolean operations and geometry validation
stay in `lib.mesh`. Placement of instances in a running scene belongs to
`RSP.simulation.scenes`.

`AssetStore(output_root)` centralizes validated output paths, category directories,
recipe keys, temporary staging and group publication with rollback. It does not
decide whether an SDF covers a radius or whether collision data are valid.
Package preparation creates a store and passes it to the asset recipes.
Generic `rigid.build` and `container.build` take a relative `stem` and a `store`.
No compatibility modules are retained at the previous import paths.

## Active call sequence and interfaces

```text
Preparation command
  → pipelines.publication.prepare_sources(support_radius=...)
    → reference recipes, Newton physical descriptions and per-link geometry checks
    → pipelines.boundary.prepare_boundary_asset + cached SDF and volume map
    → immutable published package
Playback
  → simulation.scenes.resolve.resolve_scene(..., support_radius=...)
    → published package checks and layout validation
  → Newton scene assembly
  → physics.pbf.boundary_model.load_boundaries
    → sim_ready.boundary.load.load_boundary_asset (read-only)
```

`lib.volume.pbf_volume_map.precompute_volume_map(sdf, support_radius,
quadrature_points=16)` returns an independent CUDA volume on the SDF grid.
Quadrature is keyword-only. The function performs no file IO and leaves the
input unchanged. The preparation pipeline writes the result and the input SDF
into a dual-grid `boundary.nvdb`.

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
`finalize_model` calls Newton's default `finalize()` and records its inertia
changes. Reference publication and the MimicKit backend pass
`require_dfs=True` to run the public `validate_joint_ordering()` check.
Source resource resolution and physical
structure validation remain with Newton. `describe_articulations(model)` copies
the actual `COLLIDE_SHAPES` geometry into local descriptions; runtime body IDs
are returned separately. Visual-only geometry is never substituted.
The MimicKit integration checks that each imported object has one articulation
and one joint per body in matching order. Incompatible models are reported
before runtime state construction.

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

The G1 reference recipe builds coupling assets from the native collision
primitives described in the [published package guide](PUBLISHED_ASSETS.md).

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

For publication changes, run this focused CPU contract check from the repository
root in the active environment. It uses temporary fixtures and does not bake assets.

```powershell
python -B -m unittest discover -s tests -p test_rsp_publication.py -v
```

For playback using read-only published assets, see
the [published package guide](PUBLISHED_ASSETS.md). Run checks for SDF, volume map
or collision generation only when those production paths change.
