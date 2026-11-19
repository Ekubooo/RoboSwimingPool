"""Validate the Newton model layout required by MimicKit."""

import numpy as np

from RSP.sim_ready.lib.asset_validation import ValidationReport, check_finite, check_transform


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
