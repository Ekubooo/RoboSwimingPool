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
│   ├── mesh/                         geometry, validation, convex algorithms
│   ├── volume/                       SDF baking and volume-map quadrature
│   └── storage/                      OBJ/NVDB/hull IO, fingerprints,
│                                     rollback publication, AssetStore
└── pipelines/
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
  → actual body geometry → dynamic boundary preparation with the same store
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
```

Asset fixtures and their compilation caches live under `data/assets_RSP/tests`.
`tools/check_newton_pbf_refactor.py snapshot` and `compare` check demo state,
model mappings and reset without recording video. Backend reports retain their
existing output location. The optional `integration` command still checks policy
playback and recording, but is not part of routine asset/package migration checks.
Historical preset runtime restoration and application launch normalization remain
separate tasks.
