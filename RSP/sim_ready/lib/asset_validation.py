"""Consumer checks on plain data; no simulator, importer or runtime dependency.

Newton owns physical-model structure and inertia validation. These checks cover
the additional PBF and MimicKit contracts, without repairing an input asset.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
import warnings

import numpy as np


@dataclass(frozen=True)
class Diagnostic:
    status: str
    code: str
    subject: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Diagnostic] = field(default_factory=list)

    def add(self, status, code, subject, message):
        if status not in {"pass", "warning", "error", "unsupported", "not_checked"}:
            raise ValueError(f"Unknown validation status: {status}")
        self.issues.append(Diagnostic(status, code, str(subject), str(message)))

    def extend(self, other):
        self.issues.extend(other.issues)

    @property
    def ok(self):
        return not any(i.status in {"error", "unsupported"} for i in self.issues)

    def require_valid(self):
        if not self.ok:
            raise AssetValidationError(self)

    def to_dict(self):
        return {"ok": self.ok, "issues": [asdict(i) for i in self.issues]}


class AssetValidationError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("\n".join(f"{i.subject}: [{i.code}] {i.message}"
                                  for i in report.issues if i.status in {"error", "unsupported"}))


@contextmanager
def diagnostic_scope(report, code, subject):
    """Keep the original exception as cause and both record and emit warnings."""
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                yield
            except AssetValidationError:
                raise
            except Exception as exc:
                report.add("error", code, subject, f"{type(exc).__name__}: {exc}")
                raise AssetValidationError(report) from exc
            else:
                report.add("pass", code, subject, "Completed")
    finally:
        for warning in caught:
            report.add("warning", code, subject, warning.message)
        # Emit after leaving catch_warnings, including on failure.
        for warning in caught:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)


def check_source(path, extensions, report):
    path = Path(path)
    if path.suffix.lower() not in extensions:
        report.add("unsupported", "source.format", path, f"Supported formats: {sorted(extensions)}")
        report.require_valid()
    with diagnostic_scope(report, "source.readable", path):
        with path.open("rb") as stream:
            if not stream.read(1):
                raise ValueError("Empty asset file")
    return path


@dataclass(frozen=True)
class ShapeDescription:
    name: str
    kind: str
    scale: np.ndarray
    transform: np.ndarray
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None


@dataclass(frozen=True)
class BodyDescription:
    name: str
    shapes: tuple[ShapeDescription, ...]
    mass: float = 0.
    center_of_mass: object = (0., 0., 0.)
    inertia: object = ((0., 0., 0.), (0., 0., 0.), (0., 0., 0.))


@dataclass(frozen=True)
class JointDescription:
    name: str
    parent: int
    child: int
    kind: str
    coordinates: int
    dofs: int
    dof_start: int


@dataclass(frozen=True)
class PhysicalAsset:
    name: str
    bodies: tuple[BodyDescription, ...]
    joints: tuple[JointDescription, ...] = ()
    # These are the conventions actually supplied to the importer, never inferred.
    units: str = "m, kg, s"
    up_axis: str = "Z"
    scale: tuple = (1., 1., 1.)


@dataclass(frozen=True)
class SkeletonDescription:
    body_names: tuple[str, ...]
    parents: tuple[int, ...]
    joint_types: tuple[str, ...]
    dof_starts: tuple[int, ...]
    dof_counts: tuple[int, ...]
    controlled_dofs: tuple[int, ...] = ()
    control_mode: str = "none"
    clip_physical_targets: bool = False


def check_names(names, subject, report):
    if any(not name for name in names) or len(set(names)) != len(names):
        report.add("error", "names.ambiguous", subject, f"Empty or duplicate names: {list(names)}")


def check_finite(value, subject, report):
    try:
        valid = np.isfinite(np.asarray(value)).all()
    except (TypeError, ValueError):
        valid = False
    if not valid:
        report.add("error", "numeric.finite", subject, "Expected finite numeric values")
    return bool(valid)


def check_transform(value, subject, report):
    pose = np.asarray(value)
    if not check_finite(pose, subject, report):
        return
    if pose.shape != (7,) or not np.isclose(np.linalg.norm(pose[3:]), 1., rtol=1e-5, atol=1e-6):
        report.add("error", "transform.rigid", subject, "Expected translation + unit quaternion (xyzw)")


def validate_physical_asset(asset, excluded_links=()):
    report = ValidationReport()
    check_names([body.name for body in asset.bodies], asset.name, report)
    check_names([joint.name for joint in asset.joints], asset.name + "/joints", report)
    if not asset.bodies:
        report.add("error", "bodies.empty", asset.name, "No bodies")
    scale = np.asarray(asset.scale)
    if not check_finite(scale, asset.name + "/scale", report) or scale.shape != (3,) or np.any(scale == 0):
        report.add("error", "conventions.scale", asset.name, "Expected finite, nonzero XYZ scale")
    if asset.up_axis not in {"X", "Y", "Z"} or not asset.units:
        report.add("error", "conventions", asset.name, "Units and a supported up axis must be explicit")
    if report.ok:
        report.add("pass", "conventions", asset.name,
                   f"units={asset.units}; up={asset.up_axis}; import scale={asset.scale}; body-local geometry")
    for body in asset.bodies:
        subject = f"{asset.name}/{body.name}"
        check_finite(body.mass, subject + "/mass", report)
        check_finite(body.center_of_mass, subject + "/com", report)
        check_finite(body.inertia, subject + "/inertia", report)
        if body.name in excluded_links:
            continue
        check_names([s.name for s in body.shapes], subject, report)
        for shape in body.shapes:
            label = f"{subject}/{shape.name}"
            check_transform(shape.transform, label, report)
            size = np.asarray(shape.scale)
            if not check_finite(size, label + "/scale", report):
                continue
            if size.shape != (3,):
                report.add("error", "shape.dimensions", label, "Expected three scale components")
                continue
            count = {"SPHERE": 1, "BOX": 3, "CAPSULE": 2, "CYLINDER": 2}.get(shape.kind)
            if count is not None:
                invalid_size = (size[0] <= 0 or size[1] < 0) if shape.kind == "CAPSULE" else np.any(size[:count] <= 0)
                if invalid_size:
                    report.add("error", "shape.dimensions", label, "Primitive dimensions must be positive")
                if shape.kind == "CYLINDER" and size[2] != 0:
                    report.add("unsupported", "shape.barrel", label, "Only cylinders with zero barrel radius are supported")
            elif shape.kind in {"MESH", "CONVEX_MESH"}:
                if np.any(size == 0):
                    report.add("error", "shape.scale", label, "Mesh scale must be nonzero")
                if shape.vertices is None or shape.faces is None:
                    report.add("error", "shape.source", label, "Missing collision mesh arrays")
                else:
                    vertices, faces = np.asarray(shape.vertices), np.asarray(shape.faces)
                    check_finite(vertices, label + "/vertices", report)
                    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                        report.add("error", "shape.vertices", label, "Expected nonempty XYZ vertices")
                        continue
                    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                            or not np.issubdtype(faces.dtype, np.integer)
                            or faces.min() < 0 or faces.max() >= len(vertices)):
                        report.add("error", "shape.indices", label, "Expected valid integer triangle indices")
            else:
                report.add("unsupported", "shape.type", label, shape.kind)
    if report.ok:
        report.add("pass", "physical.values", asset.name, "Names, finite values, dimensions and local transforms")
    return report


def validate_coverage(asset, prepared, excluded_links=()):
    report = ValidationReport()
    names = [body.name for body in asset.bodies]
    excluded = list(excluded_links)
    if len(set(excluded)) != len(excluded) or set(excluded) - set(names):
        report.add("error", "coverage.exclusions", asset.name, f"Duplicate or unknown exclusions: {excluded}")
    expected = {i for i, name in enumerate(names) if name not in excluded}
    actual = list(prepared)
    if len(actual) != len(set(actual)) or set(actual) != expected:
        report.add("error", "coverage.bodies", asset.name,
                   f"Expected local bodies {sorted(expected)}, received {actual}")
    for name in excluded:
        report.add("not_checked", "coverage.excluded", f"{asset.name}/{name}", "Explicitly excluded from PBF")
    if report.ok:
        report.add("pass", "coverage.bodies", asset.name, f"{len(expected)} included, {len(excluded)} excluded")
    return report


def validate_skeleton(asset, skeleton):
    """Check the consumer's existing DOF representation, not motion or policy data."""
    report = ValidationReport()
    n = len(asset.bodies)
    check_names(skeleton.body_names, "MimicKit skeleton", report)
    if not n or any(len(values) != n for values in (skeleton.body_names, skeleton.parents,
            skeleton.joint_types, skeleton.dof_starts, skeleton.dof_counts)):
        report.add("error", "skeleton.count", asset.name, "Body and per-body metadata counts differ")
        return report
    if tuple(b.name for b in asset.bodies) != skeleton.body_names:
        report.add("error", "skeleton.order", asset.name, "Body names/order differ from the loaded skeleton")
    joints = {joint.child: joint for joint in asset.joints}
    if len(joints) != n or len(asset.joints) != n or set(joints) != set(range(n)):
        report.add("unsupported", "skeleton.joints", asset.name, "MimicKit requires one joint per body")
        return report
    offset = 0
    for i in range(n):
        joint = joints[i]
        if joint.parent != skeleton.parents[i]:
            report.add("error", "skeleton.parent", asset.bodies[i].name, "Parent differs")
        kind = "ROOT" if i == 0 and joint.kind in {"FREE", "FIXED"} else joint.kind
        dofs = 0 if i == 0 else joint.dofs
        expected_start = 0 if i == 0 else joint.dof_start - joints[0].dofs
        if (kind != skeleton.joint_types[i] or dofs != skeleton.dof_counts[i]
                or expected_start != skeleton.dof_starts[i] or expected_start != offset):
            report.add("error", "skeleton.dofs", asset.bodies[i].name,
                       f"Expected {kind} at {expected_start} with {dofs} DOFs; skeleton has "
                       f"{skeleton.joint_types[i]} at {skeleton.dof_starts[i]} with {skeleton.dof_counts[i]}")
        offset += dofs
    controlled = skeleton.controlled_dofs
    if len(set(controlled)) != len(controlled) or any(
            not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= offset for i in controlled):
        report.add("error", "control.mapping", asset.name, "Duplicate or out-of-range controlled DOFs")
    if skeleton.control_mode not in {"none", "pos", "vel", "torque", "pd_explicit"}:
        report.add("unsupported", "control.mode", asset.name, skeleton.control_mode)
    scalar_required = skeleton.control_mode == "pd_explicit" or skeleton.clip_physical_targets
    if skeleton.clip_physical_targets and skeleton.control_mode not in {"pos", "pd_explicit"}:
        report.add("unsupported", "control.clipping", asset.name, "Physical target clipping requires position control")
    if scalar_required:
        scalar = {j.dof_start - joints[0].dofs for j in asset.joints if j.parent != -1 and j.dofs == 1}
        if set(controlled) - scalar:
            report.add("unsupported", "control.scalar", asset.name, "This control mode requires scalar controlled joints")
    if report.ok:
        report.add("pass", "skeleton.mapping", asset.name, f"{n} bodies, {offset} non-root DOFs; {len(controlled)} controlled")
    report.add("not_checked", "learning.data", asset.name, "Motion, policy and observations remain owned by MimicKit")
    return report
