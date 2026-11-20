"""Prepare and reuse the current generated assets for one scene."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from ..catalog import SourceCatalog, SourceIssue, digest, read_prepared_asset, scan_sources
from ..config import BakeOptions, VOLUME_MAP_ALGORITHM_VERSION, check_volume_options
from ..lib.storage.fingerprints import physical_signature, recipe_digest
from ..runtime_store import (
    AssetPreparationError, commit_runtime_asset, recover_runtime_asset,
    runtime_asset_lease, stage_runtime_asset,
)
from ..paths import ORIGIN_ROOT, RUNTIME_ROOT, runtime_root
from .identity import asset_producer, producer_snapshot


def source_identity(source, *, producer=None):
    """Geometry/model identity independent of the consumer's support radius."""
    source.model_path  # Validate raw model dependencies before replacing generated files.
    if producer is None:
        producer = producer_snapshot()
    files = {name: digest(path) for name, path in source.input_files().items()}
    terrain_digest = source.record.get("terrain_input_digest")
    if terrain_digest is not None and files[source.record["inputs"][0]] != terrain_digest:
        raise ValueError("Terrain definition changed after discovery")
    return recipe_digest(dict(
        source=source.record,
        files=files,
        producer=asset_producer(source, producer),
    ))


def build_identity(source, support_radius, *, producer=None):
    """Fingerprint a complete boundary for this consumer's support radius."""
    check_volume_options(support_radius, 16)
    if producer is None:
        producer = producer_snapshot()
    return recipe_digest(dict(source=source_identity(source, producer=producer),
                              support_radius=float(support_radius), quadrature_points=16,
                              volume_map_algorithm=VOLUME_MAP_ALGORITHM_VERSION,
                              volume_map_producer=producer["volume_map"]))


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
    return read_prepared_asset(path, source)


def _prepare(stage, source, support_radius, device, previous, *, sdf_only=False, terrain_asset=None):
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
    if source.recipe == "obj":
        from .rigid import prepare_obj
        return prepare_obj(source, store=store, device=device, support_radius=support_radius,
                           previous=previous, sdf_only=sdf_only)
    if source.is_terrain:
        from .container import prepare_terrain
        return prepare_terrain(
            store, device, support_radius, recipe=source.terrain_recipe, terrain_asset=terrain_asset,
            previous=previous,
        )
    if sdf_only:
        raise ValueError("SDF preflight is only needed for discovered OBJ assets")
    if source.container_size is not None and source.ref.category == "containers":
        bunny_source, = source.input_files().values()
        artifact, bunny, layout = container.prepare_container(
            store, device, support_radius, source.container_size, report,
            bunny_source=bunny_source, previous=previous,
        )
        save_mesh_obj(bunny, stage / "bunny.obj", digits=17)
        return dict(boundary=artifact.manifest_path.relative_to(stage).as_posix(),
                    visible_mesh="bunny.obj", hulls="collision/hulls.npz",
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


def _terrain_with_prop(source, prop):
    dependencies = {key: prop.record["files"][prop.record[key]]
                    for key in ("mesh", "hulls") if key in prop.record}
    record = dict(source.record, terrain_prop=asdict(prop.source.ref), terrain_dependency=dependencies,
                  inputs=[*source.record["inputs"], *prop.source.record["inputs"]])
    return replace(source, record=record)


def ensure_runtime_assets(refs, *, support_radius, device="cuda:0",
                          root=RUNTIME_ROOT, origin=ORIGIN_ROOT, progress=None, terrain_prop=None):
    """Return verified current assets, building only requested missing or stale slots."""
    check_volume_options(support_radius, 16)
    refs = tuple(dict.fromkeys(refs))
    try:
        root = runtime_root(root, origin=origin)
        sources = scan_sources(origin)
        producer = producer_snapshot()
    except Exception as exc:
        raise AssetPreparationError(f"Cannot discover runtime asset sources: {exc}") from exc
    ready = {}
    with runtime_asset_lease(root, reuse=True):
        terrain_ref = terrain_asset = None
        if terrain_prop is not None:
            if progress is not None:
                progress(f"Checking terrain OBJ {terrain_prop.name}…")
            try:
                selected = [ref for ref in refs if ref in sources and sources[ref].is_terrain]
                if len(selected) != 1:
                    raise ValueError("A terrain model requires one selected procedural container")
                terrain_ref, = selected
                prop_source = sources.get(terrain_prop)
                if prop_source is None or prop_source.recipe != "obj":
                    raise ValueError("The terrain model must reference a discovered OBJ")
                terrain_asset = _prepare_obj_source(
                    root, prop_source, support_radius, device, producer=producer,
                    require_sdf_coverage=False,
                )
            except Exception as exc:
                raise AssetPreparationError(f"Cannot prepare terrain OBJ {terrain_prop.relative}: {exc}") from exc

        for ref in refs:
            if progress is not None:
                progress(f"Checking {ref.category}/{ref.name}…")
            try:
                source = sources.get(ref)
                if source is None:
                    raise ValueError("No source declaration for this asset")
                terrain_dependency = terrain_asset if ref == terrain_ref else None
                if terrain_dependency is not None:
                    source = _terrain_with_prop(source, terrain_dependency)
                identity = build_identity(source, support_radius, producer=producer)
                if terrain_dependency is not None and source_identity(
                        terrain_dependency.source, producer=producer,
                ) != terrain_dependency.record["source_digest"]:
                    raise ValueError("Terrain OBJ changed after preparation")
                previous = recover_runtime_asset(root, source)
                if (previous is not None and previous.record.get("state", "ready") == "ready"
                        and previous.record["build_digest"] == identity):
                    ready[ref] = previous
                    continue
            except Exception as exc:
                raise AssetPreparationError(f"Cannot check {ref.relative}: {exc}") from exc

            if progress is not None:
                progress(f"Preparing {ref.category}/{ref.name}…")
            with stage_runtime_asset(root, source) as stage:
                try:
                    metadata = _prepare(stage, source, float(support_radius), device, previous,
                                        terrain_asset=terrain_dependency)
                    metadata = dict(metadata, state="ready", source_digest=source_identity(
                        source, producer=producer))
                    asset = write_runtime_asset(stage, source, metadata, build_digest=identity)
                    if build_identity(source, support_radius, producer=producer) != identity:
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


@dataclass
class PreparedSources:
    """Selection inputs and per-file failures, without requiring volume maps."""
    sources: SourceCatalog
    assets: dict
    errors: list


def _covers_support(asset, support_radius):
    for relative in asset.record["links"].values():
        record = json.loads(asset.file(relative).read_text(encoding="utf-8"))
        if float(record["bake_options"]["max_support_radius"]) < support_radius:
            return False
    return bool(asset.record["links"])


def _prepare_obj_source(root, source, support_radius, device, *, producer,
                        require_sdf_coverage=True):
    """Reuse or prepare one OBJ slot under the caller's runtime lease."""
    identity = source_identity(source, producer=producer)
    previous = recover_runtime_asset(root, source)
    if (previous is not None and previous.record.get("source_digest") == identity
            and (not require_sdf_coverage or previous.record.get("state", "ready") == "ready"
                 or _covers_support(previous, support_radius))):
        return previous
    with stage_runtime_asset(root, source) as stage:
        metadata = _prepare(stage, source, float(support_radius), device, previous, sdf_only=True)
        metadata = dict(metadata, state="sdf_ready", source_digest=identity)
        asset = write_runtime_asset(stage, source, metadata, build_digest=recipe_digest(
            dict(source=identity, sdf_support_radius=float(support_radius))))
        if source_identity(source, producer=producer) != identity:
            raise ValueError("Source asset changed during preparation")
        return commit_runtime_asset(root, asset)


def prepare_sources(*, support_radius, device="cuda:0", root=RUNTIME_ROOT,
                    origin=ORIGIN_ROOT, sources=None, progress=None):
    """Prepare every discovered OBJ for selection, never computing a volume map.

    A failed input is excluded along with built-in containers depending on it.
    Matching consumer-ready assets stay intact; coverage is checked at launch.
    """
    check_volume_options(support_radius, 16)
    root = runtime_root(root, origin=origin)
    discovered = scan_sources(origin) if sources is None else sources
    available = SourceCatalog()
    available.update(discovered)
    available.errors.extend(getattr(discovered, "errors", ()))
    ready, failed_inputs = {}, set()
    producer = producer_snapshot()
    with runtime_asset_lease(root, reuse=True):
        for ref, source in tuple(available.items()):
            if source.recipe != "obj":
                continue
            if progress is not None:
                progress(f"Checking OBJ {ref.name}…")
            try:
                ready[ref] = _prepare_obj_source(root, source, support_radius, device,
                                                 producer=producer)
            except Exception as exc:
                names = source.record["inputs"]
                failed_inputs.update(names)
                available.errors.append(SourceIssue(path=str(source.origin / names[0]), message=str(exc), ref=ref))
                del available[ref]
    for ref, source in tuple(available.items()):
        if failed_inputs.intersection(source.record["inputs"]):
            available.errors.append(SourceIssue(path=str(source.ref.relative),
                message="A required OBJ failed preparation", ref=ref))
            del available[ref]
    return PreparedSources(available, ready, available.errors)
