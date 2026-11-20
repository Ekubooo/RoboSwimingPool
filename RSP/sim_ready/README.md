# RSP asset preparation

The asset pipeline builds disposable files from source models and recipes.
Startup validates discovered OBJ props and prepares their collision models and
SDFs, then opens scene selection. Playback completes only the selected scene's
volume maps before creating the Newton simulation.
See the [runtime asset guide](RUNTIME_ASSETS.md) for usage.

```text
data/assets_RSP/
├── origin/
│   ├── props/                        automatically discovered Y-up OBJ files
│   ├── articulations/                original humanoid declarations and models
│   └── terrains/<name>/terrain.yaml  confirmed terrain recipes, kept as sources
└── runtime/                          generated, ignored by Git
    └── <category>/<name>/
        ├── asset.json                build fingerprint and file checksums
        ├── boundary/ or links/        per-boundary mesh, manifest and NVDB
        └── model.xml or collision/    generated MJCF for props, hulls for pools
```

The humanoid and G1 are imported from their original MJCF files. G1 also reads
its original relative mesh files and license. Pool recipes use the source bunny
mesh; primitive props and containers have built-in descriptions. New OBJ props
need no YAML: filenames identify them, and conflicting names are reported and
excluded. Each prepared source has
one directory at `category/name`. Its internal manifest validates reuse and
records how to find each generated resource.

```text
RSP/sim_ready/
├── catalog.py                        source discovery, runtime verification
├── source_geometry.py                shared source previews and dimensions
├── container_geometry.py             shared pure terrain/OBJ geometry and layout
├── terrain.py                        validated authored terrain parameters
├── runtime_store.py                  asset leases, validation and slot replacement
├── adapters/newton.py                physical model import and extraction
├── adapters/terrain.py               Newton terrain output compatibility
├── boundary/                         boundary manifest and read-only GPU load
├── lib/
│   ├── mesh/                         geometry, validation, convex hulls
│   ├── volume/                       SDF and volume-map computation
│   └── storage/                      file formats, fingerprints and safe file writes
└── pipelines/
    ├── runtime.py                    current asset generation and reuse
    ├── identity.py                   recipe and numerical-stage code fingerprints
    ├── boundary.py                   prepare one boundary per geometry
    ├── rigid.py                      OBJ normalization, inertia and rigid model
    ├── container.py                  persist shared geometry, hulls and boundaries
    ├── terrain_authoring.py          validate and atomically save a new recipe
    ├── articulation.py               validate links, share identical boundaries
    ├── collision.py                  convex collision preparation
    └── recipes/                      reference container, prop and G1 logic
```

From the repository root in the pinned WarpSim environment:

```powershell
conda run -n WarpSim python -m pip install -e . --no-deps --no-build-isolation
conda run -n WarpSim python tools/run_rsp.py
```

## Active call sequence and interfaces

The built-in `pbf_stairs_full` uses two continuous full-floor stair levels and
a central platform at 0.1, 0.2 and 0.3 m. The original `pbf_stairs` remains a
local stair patch. User-authored terrain recipes use `TerrainRecipe` and are
discovered as containers; no new runtime category or version directory is used.
`save_terrain(name, recipe)` validates geometry and saves a new source recipe;
it does not bake SDFs, volume maps or convex decompositions. Existing names cannot
be overwritten through the editor.

All new terrains fit the 3 × 2 × 5 m inner cavity with 0.3 m walls. Grid terrain
uses 6 × 4 square cells of 0.5 m. Its random heights come from Newton, but closed
boxes are reconstructed because its duplicated side vertices do not follow the
top-face displacement. Gap geometry is raised above the fixed floor so the slab
does not fill the trench. One box description drives Boolean solids and rigid
collision; walls and the lid stay invisible.

An optional OBJ uses its prepared local mesh and convex pieces under one common
transform. Stairs and gap place it on the central platform; grid always uses the
first cell at (0.25, 0.25). The combined horizontal bounds are fitted to 95% of
the placement surface and the visible base sinks 0.003 m. The container never
reruns CoACD for this dependency. Changing a prop source or its hulls invalidates
the dependent container, while its volume-map radius does not define collision
geometry. The original mesh still supplies the Boolean and fluid SDF surfaces.

`scan_sources()` discovers inputs and records per-file errors. `prepare_sources()`
prepares every discovered OBJ for selection and returns available sources,
prepared assets and failures. Invalid inputs and their dependent containers are
excluded without stopping unrelated assets. `ensure_runtime_assets()` prepares
the selected scene for a particular support radius. Application and environment
resolve the same configured runtime directory.

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

The `sdf_ready` preparation state stores one SDF in `boundary.nvdb`. Starting
the selected scene upgrades the same directory to `ready`, with the SDF and its
matching volume map in the same NVDB. Only `ready` assets may enter simulation.
Selection keeps ready assets whose source identity still matches; coverage is
rechecked at launch. An outdated slot is replaced with the new SDF preparation
state, and its map is completed only if the asset is selected for simulation.
The map is precomputed on a separate grid. A compatible persisted
SDF can be reused when the support radius changes. Otherwise the pipeline
rebuilds the SDF with adequate coverage. The boundary loader checks the grid
metadata, layout and fingerprints before handing the volumes to the solver.

Supported source models use Newton's physical geometry and inertia handling.
`finalize_model(..., require_dfs=True)` also checks the joint order for the
reference recipes and MimicKit backend. MimicKit-specific skeleton and action
validation happens in the integration layer.

OBJ props are converted from Y-up to Z-up, centered, and uniformly scaled to a
longest extent of 0.3 m. Their default density is 1000 kg/m3. The original surface
defines the visual mesh, inertia and fluid boundary; convex collision proxies
use existing CoACD preparation. The generated MJCF has one free rigid body.
No automatic repair or broader mesh-validation guarantees are added.

Runtime identities include only the relevant recipe and shared producer code.
Convex decomposition, SDF baking and volume-map generation have separate code
and dependency fingerprints. A terrain-only edit does not invalidate an
articulation or primitive asset. When an enclosing asset needs preparation,
unchanged verified SDFs, maps and convex parts can still be reused. Old caches
without stage fingerprints are regenerated on first use; this does not add
asset versions or extra runtime slots.

`pbf_stairs` uses Newton's public `Mesh.create_terrain` interface in a fixed
3 x 2 x 5 m cavity with 0.3 m walls. Its generated visual landscape excludes the
walls and lid. A separate closed mesh defines the fluid boundary; Newton boxes
and optional model hulls define rigid contacts. These share the same placement.
