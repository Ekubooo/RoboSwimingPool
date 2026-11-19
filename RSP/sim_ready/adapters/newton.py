"""Read actual Newton collision geometry; reuse its public validation unchanged."""
from dataclasses import dataclass

import newton
import numpy as np

from ..lib.asset_validation import (BodyDescription, JointDescription, PhysicalAsset,
    ShapeDescription, check_source, diagnostic_scope)


@dataclass(frozen=True)
class ArticulationBinding:
    asset: PhysicalAsset
    body_ids: tuple[int, ...]  # Runtime mapping only; never a boundary cache key.


def import_physical_asset(builder, path, *, fix_root, enable_self_collisions, report):
    source = check_source(path, {".xml", ".urdf"}, report)
    with diagnostic_scope(report, "newton.import", source):
        options = dict(floating=not fix_root, ignore_inertial_definitions=False,
                       collapse_fixed_joints=False, enable_self_collisions=enable_self_collisions)
        if source.suffix.lower() == ".xml":
            builder.add_mjcf(str(source), convert_3d_hinge_to_ball_joints=True, **options)
        else:
            builder.add_urdf(str(source), joint_ordering="dfs", **options)


def finalize_model(builder, *, device, report, require_dfs=False):
    """Use default Newton checks/corrections, plus its opt-in public DFS check."""
    with diagnostic_scope(report, "newton.finalize", "physical model"):
        model = builder.finalize(device=device, requires_grad=False)
    if require_dfs:
        with diagnostic_scope(report, "newton.joint_ordering", "physical model"):
            ordered = builder.validate_joint_ordering()
        if not ordered:
            report.add("unsupported", "newton.joint_ordering", "physical model",
                       "DFS joint order required; no automatic reorder")
    before = np.asarray(builder.body_inertia, dtype=np.float32).reshape(-1, 3, 3)
    changed = np.any(before != model.body_inertia.numpy(), axis=(1, 2))
    if changed.any():
        report.add("warning", "newton.inertia_correction", "physical model",
                   f"Newton changed inertia for bodies {np.flatnonzero(changed).tolist()}")
    report.require_valid()
    return model


def describe_articulations(model):
    """Copy geometry once; local body indices are separate from global bindings."""
    shape_body, shape_type = model.shape_body.numpy(), model.shape_type.numpy()
    scale, transforms, flags = model.shape_scale.numpy(), model.shape_transform.numpy(), model.shape_flags.numpy()
    mass, com, inertia = model.body_mass.numpy(), model.body_com.numpy(), model.body_inertia.numpy()
    parent, child, kind = model.joint_parent.numpy(), model.joint_child.numpy(), model.joint_type.numpy()
    qs, ds, starts = model.joint_q_start.numpy(), model.joint_qd_start.numpy(), model.articulation_start.numpy()
    result = []
    for art, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        joint_ids = range(int(start), int(end))
        ids = tuple(dict.fromkeys(int(child[j]) for j in joint_ids))
        local = {body_id: i for i, body_id in enumerate(ids)}
        bodies = []
        for body_id in ids:
            shapes = []
            for s in np.flatnonzero(shape_body == body_id):
                if not flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                    continue
                shape_kind = newton.GeoType(int(shape_type[s])).name
                vertices = faces = None
                if shape_kind in {"MESH", "CONVEX_MESH"}:
                    source = model.shape_source[s]
                    if source is not None:
                        vertices = np.array(source.vertices, copy=True)
                        faces = np.array(source.indices, copy=True).reshape(-1, 3)
                shapes.append(ShapeDescription(model.shape_label[s], shape_kind, scale[s].copy(),
                                                transforms[s].copy(), vertices, faces))
            bodies.append(BodyDescription(model.body_label[body_id].split("/")[-1], tuple(shapes),
                          float(mass[body_id]), com[body_id].copy(), inertia[body_id].copy()))
        joints = tuple(JointDescription(model.joint_label[j], local.get(int(parent[j]), -1),
                      local[int(child[j])], newton.JointType(int(kind[j])).name,
                      int(qs[j + 1] - qs[j]), int(ds[j + 1] - ds[j]), int(ds[j] - ds[start])) for j in joint_ids)
        label = model.articulation_label[art]
        asset = PhysicalAsset(label, tuple(bodies), joints, up_axis="XYZ"[int(model.up_axis)])
        result.append(ArticulationBinding(asset, ids))
    return tuple(result)
