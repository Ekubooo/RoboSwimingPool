"""Explicit preparation and immutable publication; never called by playback."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import yaml

from ..catalog import AssetRef, contained, digest, read_asset, physical_signature
from ..paths import ORIGIN_ROOT, PUBLISHED_ROOT, PBF_PLAY_ROOT
from ..lib.storage.store import AssetStore


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
        previous = read_asset(destination / "asset.json")
        if previous.record["recipe_digest"] != record["recipe_digest"]:
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


def prepare_sources(*, origin=ORIGIN_ROOT, published=PUBLISHED_ROOT, cache=PBF_PLAY_ROOT, device="cuda:0"):
    """Prepare only the explicitly supported reference recipes, using existing algorithms."""
    import newton
    from importlib.metadata import version
    from ..adapters.newton import import_physical_asset, finalize_model, describe_articulations
    from ..lib.asset_validation import ValidationReport
    from ..lib.storage.mesh_io import save_mesh_obj
    from ..lib.storage.hull_io import save_hulls
    from ..config import BakeOptions
    from .articulation import prepare_articulated_asset
    from .scenes import pbf_play as recipe
    from .boundary import prepare as boundary_prepare
    from RSP.runtime.fluid_clock import DEFAULT_SETTINGS
    from solvers.pbf.pbf_helper import init_scene_parameters
    from types import SimpleNamespace

    store = AssetStore(cache)
    particles = SimpleNamespace(config=DEFAULT_SETTINGS.pbf)
    init_scene_parameters(particles)
    radius = particles.smoothing_length

    def copy_boundary(artifact, stage):
        # Copy the whole family, preserving ../ references and signed manifests.
        source = artifact.manifest_path
        relative = source.relative_to(store.root)
        family = store.root / relative.parts[0] / relative.parts[1]
        target = stage / "boundaries" / family.name
        if not target.exists():
            shutil.copytree(family, target)
        return (Path("boundaries") / family.name / source.relative_to(family)).as_posix()

    result = []
    for ref, source in scan_sources(origin).items():
        inputs = {name: digest(contained(origin, name)) for name in source["inputs"]}
        code = {str(p): digest(p) for p in (Path(recipe.__file__), Path(boundary_prepare.__file__), Path(__file__))}
        identity = dict(source=source, inputs=inputs, code=code, support_radius=radius,
                        dependencies={name: version(name) for name in ("newton", "warp-lang", "trimesh")})
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

        def populate(stage):
            report = ValidationReport()
            if source["recipe"] == "reference_pool":
                if ref.category != "containers" or ref.name != "pbf_pool":
                    raise ValueError("reference_pool requires containers/pbf_pool")
                # The existing recipe owns its original mesh location and geometry.
                if Path(origin).resolve() != ORIGIN_ROOT.resolve() or source["inputs"] != ["bunny_flat_bottom.obj"]:
                    raise ValueError("Reference pool uses the existing origin/bunny_flat_bottom.obj")
                artifact, bunny, hulls = recipe.prepare_static(store, device, radius, report=report)
                boundary = copy_boundary(artifact, stage)
                save_mesh_obj(bunny, stage / "bunny.obj", digits=17)
                save_hulls(hulls, stage / "hulls.npz")
                return dict(boundary=boundary, bunny="bunny.obj", hulls="hulls.npz")
            if source["recipe"] == "reference_prop":
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
            shutil.copyfile(model_path, stage / "model.xml")
            builder = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
            builder.default_shape_cfg.mu = 1.0
            import_physical_asset(builder, model_path, fix_root=False, enable_self_collisions=True, report=report)
            model = finalize_model(builder, device=device, report=report)
            binding, = describe_articulations(model)
            prepared = prepare_articulated_asset(binding.asset, store=store, device=device,
                bake_options=BakeOptions(voxel_size=.005, max_support_radius=radius), support_radius=radius)
            links = {binding.asset.bodies[i].name: copy_boundary(artifact, stage)
                     for i, artifact in prepared.boundaries.items()}
            return dict(model="model.xml", links=links, physical_signature=physical_signature(binding.asset))

        result.append(publish_package(ref, {"recipe_digest": key}, populate, root=published))
    return result
