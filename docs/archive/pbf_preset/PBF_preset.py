"""PBF preset: build and publish assets independently of runtime assembly."""

import argparse
from dataclasses import asdict, replace
from importlib.metadata import version
import json
from pathlib import Path
import shutil

from ...lib.storage.fingerprints import file_digest, sdf_digest
from ...config import GeometryOptions, BakeOptions
from ...lib.storage.publish import publish_files
from ...lib.storage.store import AssetStore
from ...lib.storage.hull_io import load_hulls, save_hulls
from ...paths import ASSET_ROOT, ORIGIN_ROOT
from PBF.scene_data import RIGID_ASSETS


SOURCE = ORIGIN_ROOT / "bunny_flat_bottom.obj"
OUTPUT = ASSET_ROOT / "preset"
CACHE = ASSET_ROOT / "preset_cache"
BUILD_VERSION = 1
LAYOUT_VERSION = 2


def asset_names(scene):
    if scene not in ("static", "stairs-rigids"):
        raise ValueError(f"Unknown PBF scene: {scene}")
    return ("boundary/stairs_scene",) if scene == "static" else (
        "boundary/stairs_scene", *(f"rigid/stairs_{key}" for key in RIGID_ASSETS))


def _has_collision(name):
    return name == "boundary/stairs_scene" or RIGID_ASSETS[name.removeprefix("rigid/stairs_")]["kind"] == "mesh"


def build_recipe(scene, config):
    from ...lib.mesh.convex import COACD_OPTIONS, COACD_VERSION

    support = max(.05, 4. * config.particle_radius)
    geometry = {"boundary/stairs_scene": dict(kind="stairs", version=1),
                **{f"rigid/stairs_{key}": value for key, value in RIGID_ASSETS.items()}}
    selected = {name: geometry[name] for name in asset_names(scene)}
    recipe = dict(version=BUILD_VERSION, layout_version=LAYOUT_VERSION, source=file_digest(SOURCE),
        geometry=selected,
        mesh_sources={name: file_digest(ORIGIN_ROOT / description["source"])
                      for name, description in selected.items() if description["kind"] == "mesh"},
        terrain=asdict(GeometryOptions()),
        static_bake=asdict(BakeOptions(max_support_radius=support)),
        rigid_bake=asdict(BakeOptions(voxel_size=.005, max_support_radius=support)),
        decomposition=dict(backend=COACD_VERSION, options=COACD_OPTIONS),
        dependencies={name: version(name) for name in
                      ("newton", "warp-lang", "trimesh", "manifold3d", "coacd")})
    return json.loads(json.dumps(recipe))


def prepare_geometry(recipe):
    import numpy as np
    from .. import container, rigid
    from ...lib.mesh import geometry, validation

    scene, visible, static_boxes = container.prepare(SOURCE, options=GeometryOptions(**recipe["terrain"]))
    meshes = {"boundary/stairs_scene": visible}
    for name, description in recipe["geometry"].items():
        if description["kind"] == "mesh":
            source = geometry.prepare_rigid(ORIGIN_ROOT / description["source"],
                scale=description["scale"], obj_up_axis=description["obj_up_axis"])
            meshes[name] = geometry.union(validation.validate_rigid_parts(source))
        elif name != "boundary/stairs_scene":
            meshes[name] = rigid.prepare(description)
    if np.asarray(static_boxes).shape != (24, 2, 3):
        raise ValueError("The current stairs recipe requires 24 collision boxes")
    return scene, meshes, static_boxes


def validate_asset(root, name, description, device):
    """Offline content checks; the mutable VM is excluded from SDF identity."""
    import warp as wp
    import numpy as np
    import trimesh
    from ...lib.mesh.convex import validate_hulls
    from PBF.boundary_model import _read_boundary_volumes, _check_volume_map_layout

    stem = root / name
    metadata = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    if (metadata["version"] != 1 or metadata["geometry"] != description
            or metadata["local_frame"] != "asset"):
        raise ValueError(f"Asset recipe/frame differs: {stem}")
    if file_digest(stem.with_suffix(".obj")) != metadata["mesh_sha256"]:
        raise ValueError(f"Mesh digest differs: {stem}")
    mesh = trimesh.load(stem.with_suffix(".obj"), force="mesh", process=False)
    if not np.isfinite(mesh.vertices).all() or mesh.is_empty:
        raise ValueError(f"Invalid published mesh: {stem}")
    sdf, vm = _read_boundary_volumes(stem.with_suffix(".nvdb"), wp.get_device(device))
    if sdf_digest(sdf) != metadata["sdf_sha256"]:
        raise ValueError(f"SDF digest differs: {stem}")
    if vm is not None:
        _check_volume_map_layout(sdf, vm)
        if not np.array_equal(sdf.get_tiles().numpy(), vm.get_tiles().numpy()):
            raise ValueError(f"VM tile layout differs: {stem}")
    if _has_collision(name):
        path = stem.with_suffix(".collision.npz")
        if file_digest(path) != metadata["collision_sha256"]:
            raise ValueError(f"Collision digest differs: {stem}")
        validate_hulls(load_hulls(path))
    if name == "boundary/stairs_scene":
        boxes = np.asarray(metadata["static_boxes"])
        if boxes.shape != (24, 2, 3) or not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= 0):
            raise ValueError(f"Invalid static collision boxes: {stem}")
    return metadata, mesh, sdf


def asset_files(root, names):
    return [root / (name + suffix) for name in names
            for suffix in ((".obj", ".nvdb", ".json", ".collision.npz")
                           if _has_collision(name) else (".obj", ".nvdb", ".json"))]


def copy_assets(source, destination, names):
    for path in asset_files(source, names):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def reuse_published(root, stage, recipe, meshes, static_boxes, device):
    """Copy matching assets individually; a partial or corrupt group is an error."""
    reused = set()
    for name, description in recipe["geometry"].items():
        if all(path.is_file() for path in asset_files(stage, (name,))):
            continue
        files = asset_files(root, (name,))
        if not any(path.exists() for path in files):
            continue
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete published asset: {path}")
        # Validate integrity against the stored recipe before deciding whether
        # it is stale. Changing a source/recipe permits rebuilding, not corruption.
        stored_description = json.loads((root / name).with_suffix(".json").read_text(encoding="utf-8"))["geometry"]
        metadata, stored, sdf = validate_asset(root, name, stored_description, device)
        if _matches_recipe(name, description, metadata, stored, sdf, recipe, meshes[name], static_boxes):
            copy_assets(root, stage, (name,))
            reused.add(name)
    if reused:
        print(f"Reusing verified published assets: {root}: {', '.join(sorted(reused))}", flush=True)
    return reused


def _matches_recipe(name, description, metadata, stored, sdf, recipe, generated, static_boxes):
    """Check build inputs separately from published-file integrity."""
    import numpy as np
    from PBF.boundary_model import _check_sdf_support

    if metadata["geometry"] != description:
        return False
    is_static = name == "boundary/stairs_scene"
    precision = 8 if is_static else 17
    if (generated.vertices.shape != stored.vertices.shape
            or not np.allclose(np.round(generated.vertices, precision), stored.vertices,
                               rtol=0., atol=0. if is_static else 1e-15)
            or not np.array_equal(generated.faces, stored.faces)):
        return False
    build = metadata["build"]
    if (build.get("dependencies", recipe["dependencies"]) != recipe["dependencies"]
            or build.get("producer_version", 1) != BUILD_VERSION):
        return False
    bake = recipe["static_bake" if is_static else "rigid_bake"]
    if not np.allclose(sdf.get_voxel_size(), bake["voxel_size"], rtol=1e-6, atol=0.):
        return False
    if is_static:
        if (metadata["sources"] != {SOURCE.name: recipe["source"]}
                or build["geometry"] != recipe["terrain"]
                or build["decomposition"] != recipe["decomposition"]
                or not np.array_equal(metadata["static_boxes"], static_boxes)):
            return False
    elif description["kind"] == "mesh":
        if (metadata["sources"] != {description["source"]: recipe["mesh_sources"][name]}
                or build["decomposition"] != recipe["decomposition"]
                or any(build.get(key) != value for key, value in bake.items())):
            return False
    elif build.get("voxel_size") != bake["voxel_size"]:
        return False
    try:
        _check_sdf_support(sdf, bake["max_support_radius"])
    except ValueError:
        return False
    return True


def generate_assets(stage, recipe, scene, meshes, static_boxes, device):
    import numpy as np
    import warp as wp
    from .. import container, rigid

    store = AssetStore(stage)
    products = {}
    if "boundary/stairs_scene" in meshes:
        products["boundary/stairs_scene"] = container.build(scene, meshes["boundary/stairs_scene"], "boundary/stairs_scene", store=store,
            geometry_options=GeometryOptions(**recipe["terrain"]), bake_options=BakeOptions(**recipe["static_bake"]),
            collision_name="bunny", device=device)
    for name, mesh in meshes.items():
        if name == "boundary/stairs_scene":
            continue
        products[name] = rigid.build(mesh, name, store=store, bake_options=BakeOptions(**recipe["rigid_bake"]),
                                     collision=_has_collision(name),
                                     device=device)
    # The preset owns the published attachment format and the scene-wide transaction.
    for name, (mesh_path, sdf_path, hulls) in products.items():
        stem = stage / name
        is_static = name == "boundary/stairs_scene"
        bake = BakeOptions(**recipe["static_bake" if is_static else "rigid_bake"])
        description = recipe["geometry"][name]
        sources = {SOURCE.name: recipe["source"]} if is_static else (
            {description["source"]: recipe["mesh_sources"][name]} if description["kind"] == "mesh" else {})
        with sdf_path.open("rb") as stream:
            volume = wp.Volume.load_from_nvdb(stream, device=device)
        metadata = dict(version=1, geometry=recipe["geometry"][name], local_frame="asset",
            mesh_sha256=file_digest(mesh_path), sdf_sha256=sdf_digest(volume),
            sources=sources,
            build=dict(geometry=recipe["terrain"], decomposition=recipe["decomposition"],
                       bake=asdict(bake), dependencies=recipe["dependencies"]) if is_static else
                  dict(asdict(bake), decomposition=recipe["decomposition"] if _has_collision(name) else None,
                       dependencies=recipe["dependencies"]))
        if _has_collision(name):
            save_hulls(hulls, stem.with_suffix(".collision.npz"))
            metadata["collision_sha256"] = file_digest(stem.with_suffix(".collision.npz"))
        if is_static:
            metadata["static_boxes"] = static_boxes
        metadata["build"]["producer_version"] = BUILD_VERSION
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def build_scene_assets(scene, config=None, *, device="cuda:0", output_dir=None,
                       cache_dir=None, rebuild=False):
    """Prepare and validate all attachments before replacing any published file."""
    import warp as wp
    from PBF.pbf_helper import PBFConfig
    from PBF.boundary_model import load_boundary_volumes

    config = config or PBFConfig()
    recipe = build_recipe(scene, config)
    store = AssetStore(OUTPUT if output_dir is None else output_dir)
    cache_store = AssetStore(CACHE if cache_dir is None else cache_dir)
    root = store.root
    cache = cache_store.recipe_directory("scenes", recipe, name=scene, allow_nan=True)
    if root == cache or root.is_relative_to(cache) or cache.is_relative_to(root):
        raise ValueError("Published assets and the build cache must be separate directories")
    names = tuple(recipe["geometry"])
    with wp.ScopedDevice(device), cache_store.staging(cache, prefix="build-") as stage:
        if not rebuild and (cache / "recipe.json").is_file():
            if json.loads((cache / "recipe.json").read_text(encoding="utf-8")) != recipe:
                raise ValueError("Cached recipe differs; use --rebuild")
            for name, description in recipe["geometry"].items():
                validate_asset(cache, name, description, device)
            copy_assets(cache, stage, names)
            print(f"Build cache hit: {cache}", flush=True)
        else:
            prepared, meshes, boxes = prepare_geometry(recipe)
            if not rebuild:
                for candidate in dict.fromkeys((root, OUTPUT)):
                    reuse_published(candidate, stage, recipe, meshes, boxes, device)
            pending = {name: mesh for name, mesh in meshes.items()
                       if not all(path.is_file() for path in asset_files(stage, (name,)))}
            if pending:
                generate_assets(stage, recipe, prepared, pending, boxes, device)
        for name, description in recipe["geometry"].items():
            # VM precomputation is the same implementation as startup's cache repair.
            load_boundary_volumes((stage / name).with_suffix(".nvdb"),
                                  4. * config.particle_radius, device=device)
            validate_asset(stage, name, description, device)
        recipe_path = stage / "recipe.json"
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        files = asset_files(stage, names)
        targets = [(path, cache / path.relative_to(stage)) for path in files]
        targets += [(recipe_path, cache / "recipe.json")]
        targets += [(path, root / path.relative_to(stage)) for path in files]
        # Retain new decomposition diagnostics in the offline cache only.
        for path in (stage / "collision").rglob("*.json"):
            targets.append((path, cache / path.relative_to(stage)))
        publish_files(targets)
    print(f"Published {scene}: {root}", flush=True)
    return asset_files(root, names)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("static", "stairs-rigids"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--particle-radius", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--rebuild", action="store_true", help="Regenerate geometry, SDF and collision, ignoring reuse")
    args = parser.parse_args(argv)
    if args.config is not None and args.particle_radius is not None:
        parser.error("--config cannot be combined with --particle-radius")
    from PBF.pbf_helper import PBFConfig
    from PBF.config_io import load_config
    try:
        config = PBFConfig() if args.config is None else load_config(args.config)
        args.config = config if args.particle_radius is None else replace(config, particle_radius=args.particle_radius)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    del args.particle_radius
    return args


def main(argv=None):
    args = parse_args(argv)
    build_scene_assets(**vars(args))


if __name__ == "__main__":
    main()
