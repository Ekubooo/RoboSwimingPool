"""Read display geometry from asset sources without building a simulation."""
from dataclasses import dataclass
from itertools import product

import numpy as np
import trimesh


@dataclass
class PreviewGeometry:
    triangles: np.ndarray  # Expanded positions, normals and RGB, nine floats per vertex.
    lines: np.ndarray      # Expanded positions and RGB, six floats per endpoint.

    @property
    def points(self):
        return np.concatenate((self.triangles[:, :3], self.lines[:, :3]))


def _triangles(mesh, color):
    points = np.asarray(mesh.vertices)[mesh.faces].reshape(-1, 3)
    normals = np.repeat(mesh.face_normals, 3, axis=0)
    colors = np.broadcast_to(color, points.shape)
    return np.column_stack((points, normals, colors)).astype(np.float32)


def load_geometry(asset, color=None, *, prepared=None):
    """Use authored geometry and primitives without preparing boundary data."""
    lines = np.empty((0, 6), dtype=np.float32)
    if asset.container_size is not None:
        size = np.asarray(asset.container_size)
        corners = np.array(list(product((0., 1.), repeat=3))) * size
        edges = [corners[[a, b]] for a in range(8) for b in range(a+1, 8)
                 if np.count_nonzero(corners[a] != corners[b]) == 1]
        points = np.concatenate(edges)
        lines = np.column_stack((points, np.tile([.55, .72, .85], (len(points), 1)))).astype(np.float32)
        if asset.recipe == "stairs":
            from RSP.sim_ready.pipelines.container import describe_terrain
            visible = describe_terrain().visible_mesh
        else:
            visible = asset.bunny_mesh()
        return PreviewGeometry(_triangles(visible, [.72, .72, .72]), lines)

    if asset.prop_kind is not None:
        if asset.prop_kind == "box":
            mesh = trimesh.creation.box(extents=(asset.prop_size,) * 3)
        elif asset.prop_kind == "sphere":
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=asset.prop_size)
        elif asset.prop_kind == "mesh":
            mesh = (trimesh.load(prepared.file(prepared.record["mesh"]), force="mesh", process=False)
                    if prepared is not None else asset.prop_mesh())
        else:
            raise ValueError(f"Unsupported preview geometry: {asset.prop_kind}")
        return PreviewGeometry(_triangles(mesh, [.7, .7, .7] if color is None else color), lines)

    import newton
    import warp as wp

    model_path = asset.model_path
    if model_path is None:
        raise ValueError(f"Asset has no preview source: {asset.ref.relative}")
    with wp.ScopedDevice('cpu'):
        builder = newton.ModelBuilder()
        builder.add_mjcf(str(model_path), parse_sites=False,
                         parse_mujoco_options=False, enable_self_collisions=False,
                         convert_3d_hinge_to_ball_joints=True)
    meshes = []
    for i, flags in enumerate(builder.shape_flags):
        if not flags & int(newton.ShapeFlags.VISIBLE):
            continue
        kind = newton.GeoType(builder.shape_type[i])
        scale = np.asarray(builder.shape_scale[i])
        if kind == newton.GeoType.MESH:
            source = builder.shape_source[i]
            mesh = trimesh.Trimesh(source.vertices, np.asarray(source.indices).reshape(-1, 3), process=False)
            mesh.apply_scale(scale)
        elif kind == newton.GeoType.BOX:
            mesh = trimesh.creation.box(extents=2*scale)
        elif kind == newton.GeoType.SPHERE:
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=scale[0])
        elif kind == newton.GeoType.CAPSULE:
            mesh = trimesh.creation.capsule(height=2*scale[1], radius=scale[0], count=[16, 16])
        else:
            raise ValueError(f'Unsupported preview geometry: {kind.name}')
        pose = builder.shape_transform[i]
        body = builder.shape_body[i]
        if body >= 0:
            pose = wp.transform_multiply(builder.body_q[body], pose)
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(wp.quat_to_matrix(wp.transform_get_rotation(pose))).reshape(3, 3)
        matrix[:3, 3] = np.asarray(wp.transform_get_translation(pose))
        mesh.apply_transform(matrix)
        meshes.append(_triangles(mesh, builder.shape_color[i] if color is None else color))
    if not meshes:
        raise ValueError('Asset contains no visible geometry')
    return PreviewGeometry(np.concatenate(meshes), lines)


def project_geometry(geometry, aspect=4/3):
    """Right-front azimuth 45°, elevation 45°; fit all points with 10% margin."""
    toward_eye = np.array([.5, -.5, 2**-.5])
    right = np.cross([0., 0., 1.], toward_eye)
    right /= np.linalg.norm(right)
    up = np.cross(toward_eye, right)
    basis = np.array([right, up, -toward_eye])
    points = geometry.points
    center = (points.min(axis=0) + points.max(axis=0))/2
    projected = (points-center) @ basis.T
    half = np.maximum(np.max(np.abs(projected), axis=0), 1e-5)
    height = max(half[1], half[0]/aspect)*1.1
    extent = np.array([height*aspect, height, half[2]*1.1])
    def positions(data):
        return ((data[:, :3]-center) @ basis.T / extent).astype(np.float32)
    light = toward_eye + up*.5
    light /= np.linalg.norm(light)
    shade = .35 + .65*np.clip(geometry.triangles[:, 3:6] @ light, 0., 1.)
    triangles = np.column_stack((positions(geometry.triangles), geometry.triangles[:, 6:9]*shade[:, None]))
    lines = np.column_stack((positions(geometry.lines), geometry.lines[:, 3:6]))
    return triangles.astype(np.float32), lines.astype(np.float32)
