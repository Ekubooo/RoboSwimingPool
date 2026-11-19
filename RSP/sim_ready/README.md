# RSP asset preparation

The asset pipeline builds disposable files from source models and recipes.
Playback prepares the selected scene's assets before it creates the Newton
simulation. The scene selector can display source assets on a clean checkout.
See the [runtime asset guide](RUNTIME_ASSETS.md) for usage.

```text
data/assets_RSP/
├── origin/                           source manifests, models and meshes
└── runtime/                          generated, ignored by Git
    └── <category>/<name>/
        ├── asset.json                build fingerprint and file checksums
        ├── boundary/ or links/        per-boundary mesh, manifest and NVDB
        └── model.xml or collision/    generated MJCF for props, hulls for pools
```

The humanoid and G1 are imported from their original MJCF files. G1 also reads
its original relative mesh files and license. Pool recipes use the source bunny
mesh; props have programmatic primitive descriptions. Each prepared source has
one directory at `category/name`. Its internal manifest validates reuse and
records how to find each generated resource.

```text
RSP/sim_ready/
├── catalog.py                        source discovery, runtime verification
├── source_geometry.py                shared source previews and dimensions
├── adapters/newton.py                physical model import and extraction
├── boundary/                         boundary manifest and read-only GPU load
├── lib/
│   ├── mesh/                         geometry, validation, convex hulls
│   ├── volume/                       SDF and volume-map computation
│   └── storage/                      file formats and safe directory replacement
└── pipelines/
    ├── runtime.py                    current asset generation and reuse
    ├── boundary.py                   prepare one boundary per geometry
    ├── articulation.py               validate links, share identical boundaries
    ├── collision.py                  convex collision preparation
    └── recipes/                      reference container, prop and G1 logic
```

From the repository root in the pinned WarpSim environment:

```powershell
conda run -n WarpSim python -m pip install -e . --no-deps --no-build-isolation
conda run -n WarpSim python tools/run_rsp.py
```

The API is also available to source-checkout tooling:

```python
from RSP.app.config import boundary_support_radius
from RSP.sim_ready.catalog import AssetRef
from RSP.sim_ready.pipelines.runtime import ensure_runtime_assets

ref = AssetRef("props", "scene_sphere")
assets = ensure_runtime_assets([ref], support_radius=boundary_support_radius())
model = assets[ref].model_path
```

The application derives the support radius from the configured particle
radius. The asset pipeline accepts it explicitly. Scene assembly and the solver
load the generated files without preparing or modifying them. The app holds
the runtime root lease while its environment uses those files.

The storage layer builds a complete asset in a temporary sibling directory.
It verifies that directory before replacing the current one and restores the
previous directory if the replacement fails. A later invocation can recover a
complete directory after an interrupted replacement. Source inputs are outside
the generated root, and roots overlapping source inputs are rejected.

Each boundary stores one `boundary.nvdb` containing the SDF and its matching
volume map. The map is precomputed on a separate grid. A compatible persisted
SDF can be reused when the support radius changes. Otherwise the pipeline
rebuilds the SDF with adequate coverage. The boundary loader checks the grid
metadata, layout and fingerprints before handing the volumes to the solver.

Supported source models use Newton's physical geometry and inertia handling.
`finalize_model(..., require_dfs=True)` also checks the joint order for the
reference recipes and MimicKit backend. MimicKit-specific skeleton and action
validation happens in the integration layer.
