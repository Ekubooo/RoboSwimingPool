"""Prepare and reuse the current generated assets for one scene."""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from ..catalog import digest, physical_signature, read_runtime_asset, scan_sources
from ..config import BakeOptions, check_volume_options
from ..lib.storage.fingerprints import recipe_digest
from ..lib.storage.runtime_slot import (
    AssetPreparationError, commit_runtime_asset, recover_runtime_asset,
    runtime_asset_lease, stage_runtime_asset,
)
from ..paths import ORIGIN_ROOT, RUNTIME_ROOT, runtime_root


def _producer_identity():
    """Track the producer implementation independently of checkout location."""
    root = Path(__file__).resolve().parents[1]
    paths = (
        "pipelines/runtime.py", "reference.py", "source_geometry.py",
        "pipelines/recipes/container.py", "pipelines/recipes/pbf_play.py",
        "pipelines/recipes/g1.py", "config.py", "adapters/newton.py",
        "pipelines/articulation.py", "pipelines/collision.py", "pipelines/boundary.py",
        "boundary/schema.py", "lib/storage/nvdb_io.py", "lib/storage/mesh_io.py",
        "lib/storage/hull_io.py", "lib/storage/fingerprints.py",
        "lib/mesh/geometry.py", "lib/mesh/validation.py", "lib/mesh/shapes.py",
        "lib/mesh/convex.py", "lib/volume/sdf.py", "lib/volume/pbf_volume_map.py",
    )
    return {
        path: hashlib.sha256((root / path).read_text(encoding="utf-8").encode()).hexdigest()
        for path in paths
    }


def build_identity(source, support_radius, *, producer=None, dependencies=None):
    """Fingerprint one requested build, including all declared source files."""
    check_volume_options(support_radius, 16)
    source.model_path  # Validate raw model dependencies before replacing generated files.
    if producer is None:
        producer = _producer_identity()
    if dependencies is None:
        dependencies = _dependencies()
    return recipe_digest(dict(
        source=source.record,
        files={name: digest(path) for name, path in source.input_files().items()},
        support_radius=float(support_radius), producer=producer, dependencies=dependencies,
    ))


def _dependencies():
    return {name: version(name) for name in ("newton", "warp-lang", "trimesh", "manifold3d", "coacd")}


def write_runtime_asset(stage, source, metadata, *, build_digest):
    """Write and verify the internal manifest in an isolated complete asset."""
    stage = Path(stage).resolve()
    files = {}
    for path in sorted(stage.rglob("*")):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Unexpected link in staged asset: {path}")
        if path.is_file():
            files[path.relative_to(stage).as_posix()] = digest(path)
    manifest = dict(metadata, schema=1, identity=asdict(source.ref), build_digest=build_digest, files=files)
    manifest["record_sha256"] = recipe_digest(manifest)
    path = stage / "asset.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return read_runtime_asset(path, source)


def _prepare(stage, source, support_radius, device, previous):
    import newton

    from ..adapters.newton import describe_articulations, finalize_model, import_physical_asset
    from ..lib.asset_validation import ValidationReport
    from ..lib.storage.mesh_io import save_mesh_obj
    from ..lib.storage.store import AssetStore
    from .articulation import prepare_articulated_asset
    from .collision import hulls_fingerprint
    from .recipes import container, g1, pbf_play

    report = ValidationReport()
    store = AssetStore(stage)
    if source.container_size is not None and source.ref.category == "containers":
        bunny_source, = source.input_files().values()
        artifact, bunny, layout = container.prepare_container(
            store, device, support_radius, source.container_size, report,
            bunny_source=bunny_source, previous=previous,
        )
        save_mesh_obj(bunny, stage / "bunny.obj", digits=17)
        return dict(boundary=artifact.manifest_path.relative_to(stage).as_posix(),
                    bunny="bunny.obj", hulls="collision/hulls.npz",
                    hulls_fingerprint=hulls_fingerprint(bunny), container=layout)

    metadata = {}
    if source.recipe == "reference_prop" and source.ref.category == "props":
        model_path = pbf_play.write_prop_model(source.ref.name, stage / "model.xml")
        metadata["model"] = "model.xml"
    elif source.recipe in ("mimickit_g1", "mjcf") and source.ref.category == "articulations":
        model_path = source.model_path
    else:
        raise ValueError(f"Unsupported preparation recipe for {source.ref.relative}: {source.recipe}")

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 1.0
    import_physical_asset(builder, model_path, fix_root=False, enable_self_collisions=True, report=report)
    model = finalize_model(builder, device=device, report=report, require_dfs=True)
    binding, = describe_articulations(model)
    excluded = ()
    if source.recipe == "mimickit_g1":
        g1.check_collision_scope(binding.asset)
        excluded = g1.EXCLUDED_LINKS
        metadata["excluded_links"] = list(excluded)
    prepared = prepare_articulated_asset(
        binding.asset, store=store, device=device,
        bake_options=BakeOptions(voxel_size=.005, max_support_radius=support_radius),
        support_radius=support_radius, excluded_links=excluded, previous=previous,
    )
    metadata["links"] = {binding.asset.bodies[index].name: artifact.manifest_path.relative_to(stage).as_posix()
                         for index, artifact in prepared.boundaries.items()}
    metadata["physical_signature"] = physical_signature(binding.asset)
    return metadata


def ensure_runtime_assets(refs, *, support_radius, device="cuda:0",
                          root=RUNTIME_ROOT, origin=ORIGIN_ROOT, progress=None):
    """Return verified current assets, building only requested missing or stale slots."""
    check_volume_options(support_radius, 16)
    try:
        root = runtime_root(root, origin=origin)
        sources = scan_sources(origin)
        producer, dependencies = _producer_identity(), _dependencies()
    except Exception as exc:
        raise AssetPreparationError(f"Cannot discover runtime asset sources: {exc}") from exc
    ready = {}
    with runtime_asset_lease(root, reuse=True):
        for ref in dict.fromkeys(refs):
            if progress is not None:
                progress(f"Checking {ref.category}/{ref.name}…")
            try:
                source = sources.get(ref)
                if source is None:
                    raise ValueError("No source declaration for this asset")
                identity = build_identity(source, support_radius, producer=producer, dependencies=dependencies)
                previous = recover_runtime_asset(root, source)
                if previous is not None and previous.record["build_digest"] == identity:
                    ready[ref] = previous
                    continue
            except Exception as exc:
                raise AssetPreparationError(f"Cannot check {ref.relative}: {exc}") from exc

            if progress is not None:
                progress(f"Preparing {ref.category}/{ref.name}…")
            with stage_runtime_asset(root, source) as stage:
                try:
                    metadata = _prepare(stage, source, float(support_radius), device, previous)
                    asset = write_runtime_asset(stage, source, metadata, build_digest=identity)
                    if build_identity(source, support_radius, producer=producer, dependencies=dependencies) != identity:
                        raise ValueError("Source assets changed during preparation")
                except Exception as exc:
                    raise AssetPreparationError(f"Cannot prepare {ref.relative}: {exc}") from exc
                if progress is not None:
                    progress(f"Installing {ref.category}/{ref.name}…")
                try:
                    ready[ref] = commit_runtime_asset(root, asset)
                except Exception as exc:
                    raise AssetPreparationError(f"Cannot install {ref.relative}: {exc}") from exc
    return ready
