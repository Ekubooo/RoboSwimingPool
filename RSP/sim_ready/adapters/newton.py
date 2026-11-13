"""Read actual Newton collision geometry; reuse its public validation unchanged."""
from dataclasses import dataclass

import newton
import numpy as np

from ..lib.asset_validation import (BodyDescription, JointDescription, PhysicalAsset,
    ShapeDescription, ValidationReport, check_finite, check_transform, check_source, diagnostic_scope)


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


def finalize_model(builder, *, device, report):
    """Use default Newton checks/corrections, plus its opt-in public DFS check."""
    with diagnostic_scope(report, "newton.finalize", "physical model"):
        model = builder.finalize(device=device, requires_grad=False)
    with diagnostic_scope(report, "newton.joint_ordering", "MimicKit consumer"):
        ordered = builder.validate_joint_ordering()
    if not ordered:
        report.add("unsupported", "newton.joint_ordering", "MimicKit consumer", "DFS joint order required; no automatic reorder")
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


def validate_model_for_mimickit(model, bindings):
    """Consumer restrictions beyond Newton's more general valid model layouts."""
    report = ValidationReport()
    child = model.joint_child.numpy()
    if model.body_count != model.joint_count or not np.array_equal(child, np.arange(model.body_count)):
        report.add("unsupported", "consumer.body_order", "MimicKit", "Requires one joint per body in matching order")
    widths = {"FREE": (7, 6), "FIXED": (0, 0), "BALL": (4, 3), "REVOLUTE": (1, 1), "PRISMATIC": (1, 1)}
    for binding in bindings:
        for i, joint in enumerate(binding.asset.joints):
            if widths.get(joint.kind) != (joint.coordinates, joint.dofs):
                report.add("unsupported", "consumer.joint", joint.name, f"{joint.kind} coordinates/DOFs unsupported")
            if (i == 0 and (joint.parent != -1 or joint.kind not in {"FREE", "FIXED"})) or (i > 0 and joint.parent < 0):
                report.add("unsupported", "consumer.root", joint.name, "Requires a single FREE or FIXED root")
    for name in ("body_mass", "body_com", "body_inertia", "joint_q", "joint_qd", "joint_axis",
                 "joint_target_q", "joint_target_qd", "joint_target_ke", "joint_target_kd",
                 "joint_armature", "joint_damping", "joint_limit_ke", "joint_limit_kd"):
        check_finite(getattr(model, name).numpy(), name, report)
    for name in ("joint_X_p", "joint_X_c"):
        for i, pose in enumerate(getattr(model, name).numpy()):
            check_transform(pose, f"{name}/{model.joint_label[i]}", report)
    if report.ok:
        report.add("pass", "consumer.layout", "MimicKit", "Body order, joint representations and finite model arrays")
    return report
