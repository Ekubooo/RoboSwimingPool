# PBF boundary assets

The active play scene separates asset preparation from read-only runtime loading.
The existing geometry, SDF, volume integration and coupling algorithms are retained.

## Source assets and outputs

The fluid-side source meshes are `Assets/bunny_flat_bottom.obj` and
`Assets/spot_triangulated_good.obj`. They are never overwritten. MimicKit keeps
its existing model, motion and policy inputs and their existing ignore rules.
Pool geometry, primitive MJCFs, collision hulls, processed meshes, SDFs, volume
maps and manifests are generated outputs. Hand-authored configuration and build
recipes remain source files.

Repository-local output roots must be below `output/`. The play scene defaults
to `output/sim_ready/pbf_play`; external output roots are also accepted by the
tools. The source-mesh directory has a Git whitelist for its two OBJ files.

Boundary artifacts use a new `boundaries-v3` directory. Old `boundaries` caches
and existing policies are retained and are not migrated or removed. The first
run builds the new artifacts; later runs validate and reuse them.

## APIs and ownership

- `pbf_volume_map.precompute_volume_map(sdf, support_radius, quadrature_points=16)`
  computes an independent CUDA volume on the input SDF's grid. It performs no
  file IO and does not modify the SDF. `quadrature_points` is keyword-only.
- `pbf_boundary_asset.prepare_boundary_asset(mesh, checked, ...)` takes local
  geometry, its validation result, bake options, `container`, the requested
  support radius, quadrature settings, output root, device and source provenance.
  It returns a `BoundaryArtifact` referencing the published JSON manifest.
- `pbf_boundary_asset.load_boundary_asset(artifact, ...)` accepts that reference
  or its manifest path and validates the requested radius and quadrature. It
  returns GPU volumes, local bounds and a visual-mesh reference. It never writes
  or rebuilds files, and does not require the visual OBJ to exist.
- `solvers.pbf.boundary_model.load_boundaries` takes one static artifact and an
  ordered sequence of dynamic artifacts. It shares GPU resources for repeated
  references and creates independent instance transforms. The runtime owns
  body binding, pose synchronization and queries.

The play-scene pipeline chooses geometry and bake resolution. Its temporary
runtime adapter prepares static assets before Newton model finalization, then
prepares dynamic assets from the finalized bodies' collision geometry. Runtime
body IDs, masses and placements belong to `scene_manifest.json`, not cached
asset identities. Historical presets are not restored by this stage.

## Cache and publication

Within each geometry family, `sdfs/<key>` stores a single-grid SDF, processed OBJ
and `sdf.json`; `maps/<key>` stores a dual-grid `boundary.nvdb` and `boundary.json`.
References inside manifests are relative, so the artifact tree can be moved.

SDF families identify geometry, container/solid treatment, bake settings and
tool/algorithm versions. Preparation chooses the smallest existing SDF support
range covering both the requested radius and `BakeOptions.max_support_radius`.
If none is valid, it bakes a new SDF variant. Volume-map keys identify persisted
SDF contents, radius, quadrature and integration algorithm version. Changing only
quadrature or radius within the selected SDF's support range reuses the SDF.

Manifests record content fingerprints, grid transforms, actual tile coverage and
local bounding boxes/radii. Loading verifies the NVDB, manifest, SDF identity,
grid names, support range and matching SDF/map coverage. Preparation rebuilds
missing or invalid generated caches; read-only loading reports an error instead.
Staging uses the existing atomic writers and rollback-capable group publisher,
with the completed manifest published last. Construction/publication runs before
simulation; concurrent writers to the same artifact are outside the single-world
workflow supported here.

## Verification

Run `python -m unittest discover -s tests -p test_pbf_boundary_assets.py -v` in the
Newton/Warp CUDA environment. Fixtures, compilation caches and logs stay under
`output/pbf_asset_stage1_checks`. Backend regression and demo checks use
`tests/test_newton_pbf_refactor.py` and `tools/check_newton_pbf_refactor.py`.
