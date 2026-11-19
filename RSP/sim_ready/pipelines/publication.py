"""Explicit preparation and immutable publication; never called by playback."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import yaml

from ..catalog import AssetRef, contained, digest, read_asset, resolve_asset, physical_signature
from ..config import BakeOptions, check_volume_options
from ..paths import ORIGIN_ROOT, PUBLISHED_ROOT, PBF_PLAY_ROOT
from ..lib.storage.store import AssetStore
from ._legacy_publications import LEGACY_RECIPE_DIGESTS


def _producer_identity():
    """Hash producer sources under stable role names with normalized newlines."""
    root = Path(__file__).resolve().parents[1]
    files = {
        "publish": "pipelines/publication.py",
        "reference": "reference.py",
        "container_recipe": "pipelines/recipes/container.py",
        "prop_recipe": "pipelines/recipes/pbf_play.py",
        "g1_recipe": "pipelines/recipes/g1.py",
        "params": "config.py",
        "import_model": "adapters/newton.py",
        "articulation": "pipelines/articulation.py",
        "collision": "pipelines/collision.py",
        "boundary": "pipelines/boundary.py",
        "sdf": "lib/volume/sdf.py",
        "volume_map": "lib/volume/pbf_volume_map.py",
    }
    return {
        role: hashlib.sha256((root / path).read_text(encoding="utf-8").encode()).hexdigest()
        for role, path in files.items()
    }


def scan_sources(root=ORIGIN_ROOT):
    root = Path(root).resolve()
    sources = {}
    for path in sorted(root.rglob("source.yaml")):
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        if set(record) != {"schema", "identity", "recipe", "inputs"} or record["schema"] != 1:
            raise ValueError(f"Invalid source manifest: {path}")
        ref = AssetRef(**record["identity"])
        if ref in sources:
            raise ValueError(f"Duplicate source identity: {ref}")
        for relative in record["inputs"]:
            if not contained(root, relative).is_file():
                raise ValueError(f"Missing source: {relative}")
        sources[ref] = record
    return sources


def publish_package(ref, record, populate, *, root=PUBLISHED_ROOT):
    """Stage a new version completely; existing versions are never overwritten."""
    destination = contained(root, ref.relative)
    if destination.exists():
        previous = resolve_asset(ref, root)
        requested = record["recipe_digest"]
        existing = previous.record["recipe_digest"]
        if existing != requested and LEGACY_RECIPE_DIGESTS.get(requested) != existing:
            raise ValueError(f"Asset version changed: {ref.relative}; publish a new version")
        return previous
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Hidden sibling cannot match the catalog's version/asset.json scan until rename.
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir()
        metadata = populate(stage)
        files = {p.relative_to(stage).as_posix(): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
        manifest = dict(schema=1, identity=asdict(ref), **record, **metadata, files=files)
        (stage / "asset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        read_asset(stage / "asset.json")
        stage.rename(destination)
    return read_asset(destination / "asset.json")


def prepare_sources(*, support_radius, origin=ORIGIN_ROOT, published=PUBLISHED_ROOT, cache=PBF_PLAY_ROOT,
                    device="cuda:0", only=None):
    """Prepare only the explicitly supported reference recipes, using existing algorithms."""
    check_volume_options(support_radius, 16)
    radius = float(support_radius)
    import newton
    from importlib.metadata import version
    from ..adapters.newton import import_physical_asset, finalize_model, describe_articulations
    from ..lib.asset_validation import ValidationReport
    from ..lib.storage.mesh_io import save_mesh_obj
    from ..lib.storage.hull_io import save_hulls
    from .articulation import prepare_articulated_asset
    from .recipes import pbf_play as recipe

    store = AssetStore(cache)

    def copy_boundary(artifact, stage):
        # Preserve signed manifests and their relative references, copying only dependencies.
        source = artifact.manifest_path
        relative = source.relative_to(store.root)
        family = store.root / relative.parts[0] / relative.parts[1]
        def copy_file(path):
            path = path.resolve()
            if not path.is_relative_to(family.resolve()):
                raise ValueError("Boundary dependency escapes its cache family")
            target = stage / "boundaries" / family.name / path.relative_to(family.resolve())
            if target.exists():
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if path.suffix == ".json":
                for dependency in json.loads(path.read_text())["files"].values():
                    copy_file(path.parent / dependency)
        copy_file(source)
        return (Path("boundaries") / family.name / source.relative_to(family)).as_posix()

    sources = scan_sources(origin)
    if only is not None:
        if only not in sources:
            raise ValueError(f"Unknown source asset: {only}")
        sources = {only: sources[only]}
    result = []
    code = _producer_identity()
    for ref, source in sources.items():
        inputs = {name: digest(contained(origin, name)) for name in source["inputs"]}
        identity = dict(source=source, inputs=inputs, code=code, support_radius=radius,
                        dependencies={name: version(name) for name in ("newton", "warp-lang", "trimesh")})
        is_g1 = source["recipe"] == "mimickit_g1"
        if is_g1:
            from .recipes import g1 as g1_recipe
            if ref.category != "articulations" or ref.name != "g1" or source["inputs"]:
                raise ValueError("mimickit_g1 requires articulations/g1 and repository-owned inputs")
            identity["g1"] = g1_recipe.source_identity()
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

        def populate(stage):
            report = ValidationReport()
            if source["recipe"] in ("reference_pool", "wide_pool"):
                if ref.category != "containers" or ref.name not in ("pbf_pool", "pbf_pool_wide"):
                    raise ValueError("reference_pool requires containers/pbf_pool")
                # The existing recipe owns its original mesh location and geometry.
                if Path(origin).resolve() != ORIGIN_ROOT.resolve() or source["inputs"] != ["bunny_flat_bottom.obj"]:
                    raise ValueError("Reference pool uses the existing origin/bunny_flat_bottom.obj")
                from .recipes.container import prepare_container
                size = (3., 2., 5.) if source["recipe"] == "reference_pool" else (4., 2., 5.)
                artifact, bunny, hulls, layout = prepare_container(store, device, radius, size, report)
                boundary = copy_boundary(artifact, stage)
                save_mesh_obj(bunny, stage / "bunny.obj", digits=17)
                save_hulls(hulls, stage / "hulls.npz")
                return dict(boundary=boundary, bunny="bunny.obj", hulls="hulls.npz", container=layout)
            if is_g1:
                model_path = g1_recipe.copy_model(stage)
            elif source["recipe"] == "reference_prop":
                names = [spec.name for spec in recipe.RIGIDS]
                if ref.category != "props" or ref.name not in names or source["inputs"]:
                    raise ValueError("Unsupported reference prop recipe")
                model_path = Path(recipe.primitive_files(store)[names.index(ref.name)])
            elif source["recipe"] == "mjcf" and ref.category == "articulations" and len(source["inputs"]) == 1:
                model_path = contained(origin, source["inputs"][0])
                # v1 supports the self-contained current humanoid, not arbitrary imports.
                import xml.etree.ElementTree as ET
                if any("file" in element.attrib for element in ET.parse(model_path).iter()):
                    raise ValueError("External MJCF dependencies require an explicit import recipe")
            else:
                raise ValueError(f"Unsupported preparation recipe: {source['recipe']}")
            if not is_g1:
                shutil.copyfile(model_path, stage / "model.xml")
            builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            builder.default_shape_cfg.mu = 1.0
            import_physical_asset(builder, model_path, fix_root=False, enable_self_collisions=True, report=report)
            model = finalize_model(builder, device=device, report=report, require_dfs=True)
            binding, = describe_articulations(model)
            excluded = g1_recipe.EXCLUDED_LINKS if is_g1 else ()
            if is_g1:
                g1_recipe.check_collision_scope(binding.asset)
            prepared = prepare_articulated_asset(binding.asset, store=store, device=device,
                bake_options=BakeOptions(voxel_size=.005, max_support_radius=radius), support_radius=radius,
                excluded_links=excluded)
            links = {binding.asset.bodies[i].name: copy_boundary(artifact, stage)
                     for i, artifact in prepared.boundaries.items()}
            metadata = dict(model="model.xml", links=links, physical_signature=physical_signature(binding.asset))
            if is_g1:
                metadata["excluded_links"] = list(excluded)
            return metadata

        result.append(publish_package(ref, {"recipe_digest": key}, populate, root=published))
    return result
