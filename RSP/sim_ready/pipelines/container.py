"""Build the fixed pool around a closed terrain and an optional static OBJ."""

from dataclasses import asdict, dataclass
import json

import numpy as np
import trimesh

from ..config import BakeOptions
from ..lib.mesh import geometry, validation
from ..lib.mesh.convex import measure_hull_error, validate_convex_mesh, validate_hulls
from ..lib.storage.fingerprints import recipe_digest
from ..lib.storage.hull_io import load_hulls, save_hulls
from ..lib.storage.mesh_io import load_mesh_obj, save_mesh_obj
from ..reference import POOL_SIZE
from ..source_geometry import WALL_THICKNESS, normalized_obj
from .terrain_geometry import PLATFORM_WIDTH, terrain_boxes


_STAIR_SIZE = (1.6, 1.6)
_STAIR_CENTER = (2.05, 1.0)
_STEP_WIDTH = 0.15
_STEP_HEIGHT = 0.06
_MODEL_FRACTION = 0.95
_MODEL_SINK = 0.003
_SIMPLIFY_TOLERANCE = 0.001


@dataclass(frozen=True)
class TerrainDescription:
    visible_mesh: trimesh.Trimesh
    layout: dict
    model: trimesh.Trimesh | None
    hulls: tuple[trimesh.Trimesh, ...] = ()
    model_transform: np.ndarray | None = None

    @property
    def static_boxes(self):
        return self.layout["boxes"]

    @property
    def obstacle_bounds(self):
        return np.asarray(self.layout["obstacles"])


def _terrain_boxes():
    """Convert the generated stair boxes once for both rigid and fluid geometry."""
    from newton import Mesh

    terrain = Mesh.create_terrain(
        grid_size=(1, 1), block_size=_STAIR_SIZE, terrain_types="pyramid_stairs",
        terrain_params={"pyramid_stairs": dict(step_width=_STEP_WIDTH,
            step_height=_STEP_HEIGHT, platform_width=PLATFORM_WIDTH)},
        compute_inertia=False,
    )
    points = np.asarray(terrain.vertices, dtype=np.float64).copy()
    # Align generator coordinates that differ by a float32 ULP at shared faces.
    for axis in range(3):
        coordinates, inverse = np.unique(points[:, axis], return_inverse=True)
        epsilon = float(np.spacing(np.float32(max(1., np.abs(coordinates).max()))))
        anchors = coordinates.copy()
        start = 0
        for end in range(1, len(coordinates) + 1):
            if end == len(coordinates) or coordinates[end] - coordinates[start] > epsilon:
                anchors[start:end] = coordinates[start]
                start = end
        points[:, axis] = anchors[inverse]
    points[:, :2] += np.asarray(_STAIR_CENTER) - np.asarray(_STAIR_SIZE) / 2
    faces = np.asarray(terrain.indices).reshape(-1, 3)
    if not len(faces) or len(faces) % 12:
        raise ValueError("Expected twelve triangles per Newton stair box")
    boxes = []
    for block in faces.reshape(-1, 12, 3):
        ids, local = np.unique(block, return_inverse=True)
        boxes.append(trimesh.Trimesh(points[ids], local.reshape(-1, 3), process=False))
    return boxes


def _place_model(source, platform, prepared):
    if source.ref.category != "props" or source.recipe != "obj":
        raise ValueError("The platform model must be a discovered prop OBJ")
    try:
        if prepared is None:
            model, hulls = normalized_obj(source), ()
        else:
            if prepared.source.ref != source.ref:
                raise ValueError("Prepared geometry does not match the platform OBJ")
            model = load_mesh_obj(prepared.file(prepared.record["mesh"]))
            hull_path = prepared.record.get("hulls")
            if hull_path is None:
                validate_convex_mesh(model)
                hulls = (model.copy(),)
            else:
                hulls = load_hulls(prepared.file(hull_path))
                validate_hulls(hulls)

        low, high = _mesh_bounds(model, *hulls)
        span = (high - low)[:2].max()
        if span <= 0:
            raise ValueError("Expected a nonzero horizontal extent")
        scale = _MODEL_FRACTION * platform.extents[:2].min() / span
        support = np.r_[(low + high)[:2] / 2, model.bounds[0, 2]]
        target = np.r_[platform.bounds.mean(axis=0)[:2], platform.bounds[1, 2] - _MODEL_SINK]
        transform = np.diag((scale, scale, scale, 1.))
        transform[:3, 3] = target - scale * support
        for mesh in (model, *hulls):
            mesh.apply_transform(transform)
        low, high = _mesh_bounds(model, *hulls)
        if np.any(low[:2] <= 0) or low[2] <= -WALL_THICKNESS or np.any(high >= POOL_SIZE):
            raise ValueError("Placed model must fit strictly inside the container")
        return model, hulls, transform
    except (KeyError, ValueError, OSError) as error:
        raise ValueError(f"Cannot place platform OBJ {source.ref.relative}: {error}") from error


def _mesh_bounds(*meshes):
    bounds = np.asarray([mesh.bounds for mesh in meshes])
    return np.array([bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)])


def _pool_shell():
    """Return the floor and five enclosing boxes with the fixed inner dimensions."""
    x, y, z = POOL_SIZE
    w = WALL_THICKNESS
    bounds = np.array([[-w, -w, -w], [x + w, y + w, z + w]])
    # The shell boxes overlap within the walls so the Boolean keeps one cavity.
    faces = (
        (bounds[0], (x + w, y + w, 0.)),
        ((-w, -w, -w), (0., y + w, z + w)),
        ((x, -w, -w), bounds[1]),
        ((-w, -w, -w), (x + w, 0., z + w)),
        ((-w, y, -w), bounds[1]),
        ((-w, -w, z), bounds[1]),
    )
    return [trimesh.creation.box(bounds=np.asarray((low, high))) for low, high in faces], bounds


def _box_record(box):
    return dict(center=box.bounds.mean(axis=0).tolist(), half=(box.extents / 2).tolist())


def describe_terrain(terrain_source=None, *, recipe=None, terrain_asset=None):
    """Generate visible geometry and layout on CPU for previews and placement."""
    if recipe is None:
        ground, *boxes = _terrain_boxes()
        platform = max(boxes, key=lambda box: box.bounds[1, 2])
    else:
        ground, boxes, platform = terrain_boxes(recipe)
    if terrain_source is None and terrain_asset is not None:
        terrain_source = terrain_asset.source
    model, hulls, transform = (None, (), None) if terrain_source is None else _place_model(
        terrain_source, platform, terrain_asset)
    terrain_parts = [ground, *boxes]
    obstacles = [mesh.bounds.tolist() for mesh in boxes]
    if model is not None:
        terrain_parts.append(model)
        obstacles.append(_mesh_bounds(model, *hulls).tolist())
    visible = geometry.simplify(geometry.union(terrain_parts), _SIMPLIFY_TOLERANCE)
    walls = _pool_shell()[0][1:]
    x, y, z = POOL_SIZE
    probes = dict(cavity=[.5, .5, 1.], bottom=[1., 1., -.05],
                  left_wall=[-.05, 1., 1.], right_wall=[x + .05, 1., 1.],
                  front_wall=[1., -.05, 1.], back_wall=[1., y + .05, 1.],
                  ceiling=[1., 1., z + .05], platform=platform.bounds.mean(axis=0).tolist())
    layout = dict(
        units="m", up_axis="Z", size=list(POOL_SIZE), wall_thickness=WALL_THICKNESS,
        terrain="stairs" if recipe is None else recipe.kind,
        terrain_prop=None if terrain_source is None else asdict(terrain_source.ref),
        terrain_footprint=ground.bounds[:, :2].tolist(),
        boxes=[_box_record(box) for box in [*walls, *boxes]], obstacles=obstacles, probes=probes,
    )
    if recipe is not None:
        layout["terrain_recipe"] = recipe.to_dict()
    if model is not None:
        # Fluid clearance follows the original geometry; rigid clearances
        # also account for the convex approximation.
        layout["fluid_model_bounds"] = model.bounds.tolist()
    return TerrainDescription(visible, layout, model, hulls, transform)


def validate_terrain(terrain):
    """Build and validate the single closed fluid cavity before baking."""
    shell, bounds = _pool_shell()
    solid = geometry.simplify(geometry.union([*shell, terrain.visible_mesh]), _SIMPLIFY_TOLERANCE)
    checked = validation.validate_container(solid, bounds, tolerance=_SIMPLIFY_TOLERANCE)
    return solid, checked


def prepare_stairs(store, device, support_radius, *, terrain_source=None, terrain_asset=None,
                   recipe=None, previous=None):
    """Stage the visual, native collision and dual-grid fluid boundary data."""
    from ..boundary.schema import BoundaryArtifact
    from .boundary import prepare_boundary_asset

    if terrain_source is not None and terrain_asset is None:
        raise ValueError("The platform OBJ must be prepared before building the container")
    terrain = describe_terrain(terrain_source, recipe=recipe, terrain_asset=terrain_asset)
    solid, checked = validate_terrain(terrain)
    old = None if previous is None else BoundaryArtifact(previous.file(previous.record["boundary"]))
    provenance = dict(terrain=terrain.layout["terrain"], terrain_prop=terrain.layout["terrain_prop"],
                      model_fraction=_MODEL_FRACTION)
    if recipe is not None:
        provenance["terrain_recipe"] = recipe.to_dict()
    artifact = prepare_boundary_asset(
        solid, checked, store=store, device=device, support_radius=support_radius, container=True,
        bake_options=BakeOptions(voxel_size=.0075, max_support_radius=support_radius),
        provenance=provenance,
        previous=old,
    )
    save_mesh_obj(terrain.visible_mesh, store.path("landscape.obj"), digits=17)
    metadata = dict(boundary=artifact.manifest_path.relative_to(store.root).as_posix(),
                    visible_mesh="landscape.obj", container=terrain.layout)
    if terrain.hulls:
        relative = "collision/hulls.npz"
        stored = terrain_asset.record
        files = stored["files"]
        source = dict(mesh=files[stored["mesh"]], hulls=files[stored["hulls"]] if "hulls" in stored else None)
        identity = dict(source=source, transform=terrain.model_transform.tolist())
        fingerprint = recipe_digest(identity)
        metrics = dict(identity, fingerprint=fingerprint, parts=len(terrain.hulls),
                       **measure_hull_error(terrain.model, terrain.hulls))
        store.directory("collision")
        save_hulls(terrain.hulls, store.path(relative))
        store.path("collision", "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        metadata.update(hulls=relative, hulls_fingerprint=fingerprint)
    return metadata
