"""Compose fixed-pool terrain geometry for assembly, previews and baking."""

from dataclasses import asdict, dataclass

import numpy as np
import trimesh

from .adapters.terrain import legacy_stair_boxes, random_grid_cells
from .lib.mesh import geometry, validation
from .lib.mesh.convex import validate_convex_mesh, validate_hulls
from .lib.storage.hull_io import load_hulls
from .lib.storage.mesh_io import load_mesh_obj
from .reference import POOL_SIZE
from .source_geometry import WALL_THICKNESS, normalized_obj


_STAIR_SIZE = (1.6, 1.6)
_STAIR_CENTER = (2.05, 1.0)
_LEGACY_STEP_WIDTH = 0.15
_LEGACY_STEP_HEIGHT = 0.06
PLATFORM_MODEL_FRACTION = 0.95
_MODEL_SINK = 0.003
_SIMPLIFY_TOLERANCE = 0.001
_PLATFORM_WIDTH = .7
_BASE_Z = -.1
_STEP_WIDTH = .325
_CELL_SIZE = .5


@dataclass(frozen=True)
class TerrainDescription:
    visible_mesh: trimesh.Trimesh
    layout: dict
    model: trimesh.Trimesh | None
    hulls: tuple[trimesh.Trimesh, ...] = ()
    model_transform: np.ndarray | None = None

    @property
    def obstacle_bounds(self):
        return np.asarray(self.layout["obstacles"])


def _box(x0, x1, y0, y1, top):
    return trimesh.creation.box(bounds=((x0, y0, _BASE_Z), (x1, y1, top)))


def _platform(width, depth, top):
    half = _PLATFORM_WIDTH / 2
    return _box(width / 2 - half, width / 2 + half, depth / 2 - half, depth / 2 + half, top)


def _stairs(width, depth, height):
    w = _STEP_WIDTH
    platform = _platform(width, depth, 3 * height)
    # Nested rectangles fill the otherwise low strips flanking the square platform.
    return (_box(0., width, 0., depth, height),
            _box(w, width - w, w, depth - w, 2 * height), platform), platform


def _gap(width, depth, gap_width, height):
    # Match Newton's border and central island layout with the pool floor
    # as the trench bottom. All five raised boxes overlap the existing slab.
    span = _PLATFORM_WIDTH + 2 * gap_width
    x, y = (width - span) / 2, (depth - span) / 2
    if min(x, y) <= 0.:
        raise ValueError("Gap and platform must leave a positive border")
    platform = _platform(width, depth, height)
    boxes = (_box(0., width, 0., y, height),
             _box(0., width, depth - y, depth, height),
             _box(0., x, y, depth - y, height),
             _box(width - x, width, y, depth - y, height), platform)
    return boxes, platform


def _procedural_boxes(recipe):
    """Return the visible base, raised solids, and one fixed placement surface."""
    width, depth = POOL_SIZE[:2]
    params = recipe.parameters
    if recipe.kind == "pyramid_stairs":
        boxes, platform = _stairs(width, depth, params["step_height"])
    elif recipe.kind == "random_grid":
        cells = random_grid_cells(width, depth, _CELL_SIZE, params["max_height"], params["seed"])
        boxes = tuple(_box(*cell) for cell in cells)
        platform = boxes[0]
    else:
        boxes, platform = _gap(width, depth, params["gap_width"], params["depth"])
    return _box(0., width, 0., depth, 0.), boxes, platform


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
        scale = PLATFORM_MODEL_FRACTION * platform.extents[:2].min() / span
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
        ground, *boxes = legacy_stair_boxes(
            _STAIR_SIZE, _STAIR_CENTER, step_width=_LEGACY_STEP_WIDTH,
            step_height=_LEGACY_STEP_HEIGHT, platform_width=_PLATFORM_WIDTH,
        )
        platform = max(boxes, key=lambda box: box.bounds[1, 2])
    else:
        ground, boxes, platform = _procedural_boxes(recipe)
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
