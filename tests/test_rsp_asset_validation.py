"""Plain-data checks, actual Newton validation and existing skeleton reuse."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import newton
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/assets_RSP/tests/articulation"
(OUTPUT / "tmp").mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(OUTPUT / "tmp")
wp.config.kernel_cache_dir = str(OUTPUT / "warp_cache")
sys.path.insert(0, str(ROOT / "mimickit"))

from RSP.sim_ready.lib.asset_validation import (AssetValidationError, BodyDescription,
    PhysicalAsset, ShapeDescription, SkeletonDescription, ValidationReport, check_source,
    diagnostic_scope, validate_physical_asset, validate_coverage, validate_skeleton)
from RSP.sim_ready.adapters.newton import finalize_model, describe_articulations, validate_model_for_mimickit


def import_model(relative):
    builder = newton.ModelBuilder()
    report = ValidationReport()
    path = check_source(ROOT / relative, {".xml", ".urdf"}, report)
    with diagnostic_scope(report, "newton.import", path):
        builder.add_mjcf(str(path), floating=True, ignore_inertial_definitions=False,
            collapse_fixed_joints=False, enable_self_collisions=True,
            convert_3d_hinge_to_ball_joints=True)
    model = finalize_model(builder, device="cpu", report=report)
    return model, describe_articulations(model), report


def skeleton_from_existing(relative):
    from anim.mjcf_char_model import MJCFCharModel
    kin = MJCFCharModel("cpu")
    kin.load(str(ROOT / relative))
    kinds = {"HINGE": "REVOLUTE", "SPHERICAL": "BALL", "FIXED": "FIXED"}
    n = kin.get_num_joints()
    return SkeletonDescription(tuple(kin.get_body_names()), tuple(int(kin.get_parent_id(i)) for i in range(n)),
        ("ROOT",) + tuple(kinds[kin.get_joint(i).joint_type.name] for i in range(1, n)),
        (0,) + tuple(kin.get_joint_dof_idx(i) for i in range(1, n)),
        (0,) + tuple(kin.get_joint_dof_dim(i) for i in range(1, n)),
        tuple(range(kin.get_dof_size())), "pos")


def box_asset():
    shape = ShapeDescription("box", "BOX", np.array([.1, .2, .3], dtype=np.float32),
                             np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float32))
    return PhysicalAsset("fixture", (BodyDescription("link", (shape,), mass=1.),))


class AssetValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.humanoid, cls.bindings, cls.report = import_model("data/assets/humanoid/humanoid.xml")
        cls.skeleton = skeleton_from_existing("data/assets/humanoid/humanoid.xml")

    def test_existing_humanoid_and_g1_use_newton_and_loaded_skeleton(self):
        for path in ("data/assets/humanoid/humanoid.xml", "data/assets/g1/g1.xml"):
            with self.subTest(path=path):
                model, bindings, report = import_model(path)
                self.assertTrue(report.ok)
                self.assertTrue(validate_model_for_mimickit(model, bindings).ok)
                self.assertTrue(validate_physical_asset(bindings[0].asset).ok)
                result = validate_skeleton(bindings[0].asset, skeleton_from_existing(path))
                self.assertTrue(result.ok, result.to_dict())
                self.assertTrue(any(i.status == "not_checked" for i in result.issues))

    def test_input_file_and_backend_reference_errors_keep_cause(self):
        with self.assertRaises(AssetValidationError) as error:
            check_source(OUTPUT / "missing.xml", {".xml"}, ValidationReport())
        self.assertIsInstance(error.exception.__cause__, FileNotFoundError)
        with self.assertRaisesRegex(AssetValidationError, "source.format"):
            check_source("visual.fbx", {".xml"}, ValidationReport())
        builder = newton.ModelBuilder()
        link = builder.add_body()
        builder.add_shape_box(link, hx=.1, hy=.1, hz=.1)
        builder.shape_body[0] = 123
        with self.assertRaises(AssetValidationError) as error:
            finalize_model(builder, device="cpu", report=ValidationReport())
        self.assertIsNotNone(error.exception.__cause__)
        self.assertIn("newton.finalize", str(error.exception))

    def test_missing_mesh_reference_is_located_by_newton(self):
        path = OUTPUT / "missing_mesh.xml"
        path.write_text('<mujoco><asset><mesh name="absent" file="does_not_exist.obj"/></asset>'
                        '<worldbody><body><geom type="mesh" mesh="absent"/></body></worldbody></mujoco>')
        with self.assertRaises(AssetValidationError) as error:
            with diagnostic_scope(ValidationReport(), "newton.import", path):
                newton.ModelBuilder().add_mjcf(str(path))
        self.assertIn(str(path), str(error.exception))
        self.assertIsNotNone(error.exception.__cause__)

    def test_values_names_shapes_and_transforms(self):
        asset = box_asset()
        self.assertTrue(validate_physical_asset(asset).ok)
        body, shape = asset.bodies[0], asset.bodies[0].shapes[0]
        for bad in (replace(asset, bodies=(body, body)),
                    replace(asset, bodies=(replace(body, mass=np.nan),)),
                    replace(asset, bodies=(replace(body, shapes=(replace(shape, scale=np.zeros(3)),)),)),
                    replace(asset, bodies=(replace(body, shapes=(replace(shape, transform=np.zeros(7)),)),)),
                    replace(asset, bodies=(replace(body, shapes=(replace(shape, kind="CONE"),)),))):
            self.assertFalse(validate_physical_asset(bad).ok)

    def test_mapping_rejects_body_order_parent_and_dof_mismatch(self):
        asset, kin = self.bindings[0].asset, self.skeleton
        cases = (replace(kin, body_names=tuple(reversed(kin.body_names))),
                 replace(kin, parents=(-1,) * len(kin.parents)),
                 replace(kin, dof_counts=(0,) * len(kin.parents)),
                 replace(kin, dof_starts=(0,) * len(kin.parents)),
                 replace(kin, controlled_dofs=(0, 0)),
                 replace(kin, controlled_dofs=(999,)),
                 replace(kin, control_mode="pd_explicit"))
        for bad in cases:
            self.assertFalse(validate_skeleton(asset, bad).ok)
        # Passive and fixed joints do not need actuators.
        self.assertTrue(validate_skeleton(asset, replace(kin, controlled_dofs=(), control_mode="none")).ok)

    def test_coverage_and_exclusions_are_explicit(self):
        asset = box_asset()
        self.assertTrue(validate_coverage(asset, [0]).ok)
        for selected, excluded in (([], []), ([0, 0], []), ([], ["missing"]), ([], ["link", "link"])):
            self.assertFalse(validate_coverage(asset, selected, excluded).ok)
        self.assertTrue(validate_coverage(asset, [], ["link"]).ok)

    def test_malformed_mesh_arrays_and_non_numeric_values_report_errors(self):
        asset = box_asset()
        body, shape = asset.bodies[0], asset.bodies[0].shapes[0]
        mesh = replace(shape, kind="MESH", vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
        for bad in (replace(mesh, scale=np.array([1., 0., 1.])),
                    replace(mesh, vertices=np.array(1.)),
                    replace(mesh, vertices=np.full((3, 3), np.nan)),
                    replace(mesh, faces=np.array([[0., 1., 2.]])),
                    replace(mesh, scale=np.array(["bad"] * 3)),
                    replace(mesh, transform=np.array(["bad"] * 7))):
            result = validate_physical_asset(replace(asset, bodies=(replace(body, shapes=(bad,)),)))
            self.assertFalse(result.ok)
            self.assertTrue(any(i.subject.endswith("/box") or "/box/" in i.subject for i in result.issues if i.status == "error"))

    def test_warnings_are_recorded_and_reemitted(self):
        report = ValidationReport()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with diagnostic_scope(report, "backend", "fixture"):
                warnings.warn("original correction", RuntimeWarning)
        self.assertEqual(len(caught), 1)
        self.assertTrue(any(i.status == "warning" and i.message == "original correction" for i in report.issues))

    def test_public_ordering_check_and_default_inertia_correction(self):
        from unittest.mock import patch
        builder = newton.ModelBuilder()
        link = builder.add_link(mass=1., inertia=wp.mat33(0.))
        builder.add_articulation([builder.add_joint_free(link)])
        report = ValidationReport()
        model = finalize_model(builder, device="cpu", report=report)
        self.assertTrue(np.isfinite(model.body_inertia.numpy()).all())
        self.assertTrue(any(i.code == "newton.inertia_correction" for i in report.issues))
        with patch.object(builder, "validate_joint_ordering", return_value=False) as ordering:
            with self.assertRaisesRegex(AssetValidationError, "no automatic reorder"):
                finalize_model(builder, device="cpu", report=ValidationReport())
            ordering.assert_called_once_with()

    def test_urdf_public_import_and_visual_shape_filter(self):
        from unittest.mock import patch
        from RSP.sim_ready.adapters.newton import import_physical_asset
        path = OUTPUT / "simple.urdf"
        path.write_text('<robot name="simple"><link name="link">'
                        '<inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
                        '<visual><geometry><sphere radius="0.5"/></geometry></visual>'
                        '<collision><geometry><box size="0.2 0.3 0.4"/></geometry></collision>'
                        '</link></robot>')
        builder = newton.ModelBuilder()
        report = ValidationReport()
        import_physical_asset(builder, path, fix_root=False, enable_self_collisions=True, report=report)
        model = finalize_model(builder, device="cpu", report=report)
        binding, = describe_articulations(model)
        self.assertEqual([s.kind for s in binding.asset.bodies[0].shapes], ["BOX"])
        self.assertTrue(validate_model_for_mimickit(model, (binding,)).ok)
        with patch.object(model, "up_axis", 1):
            self.assertEqual(describe_articulations(model)[0].asset.up_axis, "Y")

    def test_preflight_rejects_mapping_before_runtime_state_is_created(self):
        from unittest.mock import Mock, patch
        from engines.newton_pbf_engine import SceneBuilder
        from engines.newton_pbf_runtime import PBFScene
        with tempfile.TemporaryDirectory(dir=OUTPUT / "tmp") as directory:
            scene = PBFScene(directory, "cpu", 1 / 30, 8)
            scene.register_character(0, replace(self.skeleton, body_names=tuple(reversed(self.skeleton.body_names))))
            sentinel = scene.store.path("scene_manifest.json")
            sentinel.write_text("previous successful run")
            builder = SceneBuilder.__new__(SceneBuilder)
            builder.device = "cpu"
            builder.play_scene = scene
            builder.builder = Mock(shape_flags=[])
            builder.validate_envs = Mock()
            builder.build_objects = Mock()
            with patch.object(scene, "add_static"), patch("RSP.sim_ready.adapters.newton.finalize_model", return_value=self.humanoid):
                with self.assertRaisesRegex(AssetValidationError, "skeleton.order"):
                    builder.finalize()
            self.assertEqual(sentinel.read_text(), "previous successful run")

    def test_multiple_articulations_cannot_silently_shift_object_ids(self):
        from types import SimpleNamespace
        from engines.newton_pbf_engine import SceneBuilder
        path = OUTPUT / "two_objects.xml"
        path.write_text('<mujoco><worldbody><body name="one"><freejoint/><geom type="sphere" size=".1"/></body>'
                        '<body name="two"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        scene = SceneBuilder.__new__(SceneBuilder)
        scene._builder_cache = {}
        scene.play_scene = SimpleNamespace(validation=ValidationReport())
        with self.assertRaisesRegex(AssetValidationError, "one articulation per object; imported 2"):
            scene._create_obj_builder(str(path), False, False, True)

    def test_nonfinite_control_parameters_are_rejected(self):
        from unittest.mock import patch
        values = self.humanoid.joint_damping.numpy().copy()
        values[6] = np.nan
        bad = wp.array(values, dtype=wp.float32, device="cpu")
        with patch.object(self.humanoid, "joint_damping", bad):
            result = validate_model_for_mimickit(self.humanoid, self.bindings)
        self.assertFalse(result.ok)
        self.assertTrue(any(i.subject == "joint_damping" and i.code == "numeric.finite" for i in result.issues))


if __name__ == "__main__":
    unittest.main()
