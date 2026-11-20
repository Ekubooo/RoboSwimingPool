"""Build one fixed pool with Newton stairs and an optional platform model."""

from dataclasses import asdict, dataclass

import numpy as np
import trimesh

from ..config import BakeOptions
from ..lib.mesh import geometry, validation
from ..lib.storage.mesh_io import save_mesh_obj
from ..reference import POOL_SIZE
from ..source_geometry import WALL_THICKNESS


_STAIR_SIZE = (1.6, 1.6)
_STAIR_CENTER = (2.05, 1.0)
_STEP_WIDTH = 0.15
_STEP_HEIGHT = 0.06
_PLATFORM_WIDTH = 0.7
_MODEL_FRACTION = 0.95
_MODEL_SINK = 0.003
_SIMPLIFY_TOLERANCE = 0.001


@dataclass(frozen=True)
class TerrainDescription:
    visible_mesh: trimesh.Trimesh
    layout: dict
    model: trimesh.Trimesh | None

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
            step_height=_STEP_HEIGHT, platform_width=_PLATFORM_WIDTH)},
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


def _place_model(source, platform):
    if source.ref.category != "props" or source.recipe != "obj":
        raise ValueError("The platform model must be a discovered prop OBJ")
    path, = source.input_files().values()
    try:
        parts = validation.validate_rigid_parts(geometry.prepare_rigid(path))
        model = parts[0] if len(parts) == 1 else geometry.union(parts)
        span = model.extents[[0, 2]].max()  # Imported OBJ is Y-up.
        if span <= 0:
            raise ValueError("Expected a nonzero horizontal extent")
        model.apply_scale(_MODEL_FRACTION * platform.extents[:2].min() / span)
        # Measure the support section after scaling so the sink depth is in meters.
        support = geometry.support_center(model, _MODEL_SINK)
        transform = np.array(((1., 0., 0., 0.), (0., 0., -1., 0.),
                              (0., 1., 0., 0.), (0., 0., 0., 1.)))
        target = np.r_[platform.bounds.mean(axis=0)[:2], platform.bounds[1, 2] - _MODEL_SINK]
        transform[:3, 3] = target - transform[:3, :3] @ support
        model.apply_transform(transform)
        if np.any(model.bounds[0] <= 0) or np.any(model.bounds[1] >= POOL_SIZE):
            raise ValueError("Placed model must fit strictly inside the container")
        return model
    except (ValueError, OSError) as error:
        raise ValueError(f"Cannot place platform OBJ {path}: {error}") from error


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


def describe_terrain(terrain_source=None):
    """Generate visible geometry and layout on CPU for previews and placement."""
    ground, *steps = _terrain_boxes()
    platform = max(steps, key=lambda box: box.bounds[1, 2])
    model = None if terrain_source is None else _place_model(terrain_source, platform)
    terrain_parts = [ground, *steps]
    obstacles = steps
    if model is not None:
        terrain_parts.append(model)
        obstacles = [*steps, model]
    visible = geometry.simplify(geometry.union(terrain_parts), _SIMPLIFY_TOLERANCE)
    walls = _pool_shell()[0][1:]
    x, y, z = POOL_SIZE
    probes = dict(cavity=[.5, .5, 1.], bottom=[1., 1., -.05],
                  left_wall=[-.05, 1., 1.], right_wall=[x + .05, 1., 1.],
                  front_wall=[1., -.05, 1.], back_wall=[1., y + .05, 1.],
                  ceiling=[1., 1., z + .05], platform=platform.bounds.mean(axis=0).tolist())
    layout = dict(
        units="m", up_axis="Z", size=list(POOL_SIZE), wall_thickness=WALL_THICKNESS,
        terrain="stairs", terrain_prop=None if terrain_source is None else asdict(terrain_source.ref),
        terrain_footprint=ground.bounds[:, :2].tolist(),
        boxes=[_box_record(box) for box in [*walls, *steps]],
        obstacles=[mesh.bounds.tolist() for mesh in obstacles], probes=probes,
    )
    return TerrainDescription(visible, layout, model)


def prepare_stairs(store, device, support_radius, *, terrain_source=None, previous=None):
    """Stage the visual, native collision and dual-grid fluid boundary data."""
    from ..boundary.schema import BoundaryArtifact
    from .boundary import prepare_boundary_asset
    from .collision import hulls_fingerprint, prepare_hulls

    terrain = describe_terrain(terrain_source)
    shell, bounds = _pool_shell()
    solid = geometry.simplify(geometry.union([*shell, terrain.visible_mesh]), _SIMPLIFY_TOLERANCE)
    checked = validation.validate_container(solid, bounds, tolerance=_SIMPLIFY_TOLERANCE)
    old = None if previous is None else BoundaryArtifact(previous.file(previous.record["boundary"]))
    artifact = prepare_boundary_asset(
        solid, checked, store=store, device=device, support_radius=support_radius, container=True,
        bake_options=BakeOptions(voxel_size=.0075, max_support_radius=support_radius),
        provenance=dict(terrain="stairs", terrain_prop=terrain.layout["terrain_prop"],
                        platform_span=_PLATFORM_WIDTH, model_fraction=_MODEL_FRACTION),
        previous=old,
    )
    save_mesh_obj(terrain.visible_mesh, store.path("landscape.obj"), digits=17)
    metadata = dict(boundary=artifact.manifest_path.relative_to(store.root).as_posix(),
                    visible_mesh="landscape.obj", container=terrain.layout)
    if terrain.model is not None:
        hulls = prepare_hulls(terrain.model, "stairs_platform_model", store, destination="collision", previous=previous)
        # Fluid clearance follows the actual SDF geometry. Rigid layout also accounts for its proxies.
        terrain.layout["fluid_model_bounds"] = terrain.model.bounds.tolist()
        bounds = np.asarray([mesh.bounds for mesh in (terrain.model, *hulls)])
        terrain.layout["obstacles"][-1] = [bounds[:, 0].min(axis=0).tolist(), bounds[:, 1].max(axis=0).tolist()]
        metadata.update(hulls="collision/hulls.npz", hulls_fingerprint=hulls_fingerprint(terrain.model))
    return metadata
