"""Collision geometry equivalence and actual articulated SDF/volume-map builds."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import newton
import numpy as np
import trimesh
import warp as wp

from test_rsp_asset_validation import OUTPUT, box_asset, import_model
from RSP.sim_ready.adapters.newton import describe_articulations, finalize_model
from RSP.sim_ready.lib.asset_validation import (AssetValidationError, BodyDescription, PhysicalAsset,
    ShapeDescription, ValidationReport)
from RSP.sim_ready.lib.mesh.shapes import shape_mesh, body_mesh
from RSP.sim_ready.lib.mesh.convex import validate_convex_mesh, validate_hulls
from RSP.sim_ready.lib.storage.fingerprints import file_digest, mesh_digest
from RSP.sim_ready.lib.storage.store import AssetStore
from RSP.sim_ready.config import BakeOptions
from RSP.sim_ready.pipelines.articulation import prepare_articulated_asset
from RSP.sim_ready.pipelines.boundary.load import load_boundary_asset


def file_snapshot(root):
    return {str(p.relative_to(root)): (file_digest(p), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


class ArticulatedGeometry(unittest.TestCase):
    def test_primitives_match_original_algorithm_and_saved_baseline(self):
        model, bindings, _ = import_model("data/assets/humanoid/humanoid.xml")
        # Captured BEFORE extraction, under the repository's pinned dependencies.
        # Keep a small independent golden even when the ignored NPZ is absent.
        fingerprints = (
            "bacc810a59d3abcaa0ff2e7197a19a1b6b0c1741cb5d407a13988985d71febda",
            "a504875e9e6d20ee45c3a1799906e1552dfccf3d2665a76ba4c104f40c93e711",
            "dca330dea39b720301716180f9d5b723a407a73c421e51be0108d47736b424ea",
            "a9537b9a271c9726a606ca01b847f620d112d25bfb40ef28fe7a6733d198d656",
            "757966864a876ca402174806680d4147c6723aaba9527a47a1ecb0913b7929ea",
            "5251026fab440b3299766b7d157ad65e1cbbba70c66446483b69a2de8c56f1a5",
            "96a6b01d7f43fd5122f323c7d14a2277da84c87f6d4217e56451d42aa5ef8a12",
            "da737fe06bb937c4b7170e141f0ab76f015f8f8d9ead935b020e8793144724b7",
            "5251026fab440b3299766b7d157ad65e1cbbba70c66446483b69a2de8c56f1a5",
            "78c0e792e167b0e6201080734119fe9902e24f3337c9a44b2a01aac8afe08749",
            "5683d956674960930466b8a76b816e552afa3425a5622395f64fe53fe9a6c16f",
            "476aab67863ab466833412926e5845ce476a510ca3ab7acdec3d70ae632da72e",
            "78c0e792e167b0e6201080734119fe9902e24f3337c9a44b2a01aac8afe08749",
            "5683d956674960930466b8a76b816e552afa3425a5622395f64fe53fe9a6c16f",
            "476aab67863ab466833412926e5845ce476a510ca3ab7acdec3d70ae632da72e",
        )
        self.assertEqual(model.body_count, len(fingerprints))
        source = OUTPUT / "humanoid_geometry_pre.npz"
        saved = np.load(source) if source.is_file() else None
        exact = 0
        for index, body in enumerate(bindings[0].asset.bodies):
            mesh, checked = body_mesh(body)
            self.assertEqual(mesh_digest(mesh), fingerprints[index])
            if saved is not None:
                for name, value in (("vertices", mesh.vertices), ("faces", mesh.faces),
                                    ("checked_vertices", checked.vertices), ("checked_faces", checked.faces)):
                    np.testing.assert_array_equal(value, saved[f"{index}_{name}"])
                    exact += 1
        if saved is not None:
            saved.close()
            self.assertEqual(exact, 60)

    def test_mesh_scaling_reflection_local_transform_and_input_immutability(self):
        source = trimesh.creation.box(extents=[.2, .3, .4])
        vertices, faces = source.vertices.copy(), source.faces.copy()
        pose = np.array([.7, -.3, .4, 0., 0., np.sqrt(.5), np.sqrt(.5)])
        shape = ShapeDescription("scaled_mesh", "MESH", np.array([-2., 3., .5]), pose, vertices, faces)
        result = shape_mesh(shape)
        self.assertGreater(result.volume, 0)
        self.assertAlmostEqual(result.volume, source.volume * 3)
        expected = source.vertices * shape.scale
        expected = np.column_stack((-expected[:, 1], expected[:, 0], expected[:, 2])) + pose[:3]
        np.testing.assert_allclose(result.bounds, [expected.min(0), expected.max(0)], atol=1e-12)
        np.testing.assert_array_equal(vertices, source.vertices)
        np.testing.assert_array_equal(faces, source.faces)
        np.testing.assert_array_equal(shape.scale, [-2., 3., .5])

    def test_multi_shape_union_and_error_names(self):
        body = box_asset().bodies[0]
        second = replace(body.shapes[0], name="second", transform=np.array([.1, 0, 0, 0, 0, 0, 1.]))
        mesh, _ = body_mesh(replace(body, shapes=body.shapes + (second,)))
        self.assertAlmostEqual(mesh.volume, .3 * .4 * .6, places=7)
        source = trimesh.creation.box()
        broken = ShapeDescription("broken_triangle_mesh", "MESH", np.ones(3), second.transform,
                                  source.vertices, source.faces[:-1])
        with self.assertRaisesRegex(AssetValidationError, "link/broken_triangle_mesh") as error:
            body_mesh(replace(body, shapes=(broken,)))
        self.assertIsNotNone(error.exception.__cause__)

    def test_convex_validation_is_not_limited_by_coacd_output_budgets(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        validate_convex_mesh(mesh)
        with self.assertRaisesRegex(ValueError, "CoACD produced an invalid convex solid"):
            validate_hulls([mesh])

    def test_no_baking_until_all_geometry_and_coverage_pass(self):
        good = box_asset().bodies[0]
        empty = BodyDescription("auxiliary", ())
        asset = PhysicalAsset("invalid", (good, empty))
        with tempfile.TemporaryDirectory(dir=OUTPUT / "tmp") as directory:
            store = AssetStore(directory)
            with patch("RSP.sim_ready.pipelines.articulation.prepare_boundary_asset") as bake:
                with self.assertRaisesRegex(AssetValidationError, "auxiliary.*geometry.missing"):
                    prepare_articulated_asset(asset, store=store, bake_options=BakeOptions(), support_radius=.06)
                bake.assert_not_called()
                self.assertEqual(list(store.root.iterdir()), [])
                with self.assertRaisesRegex(AssetValidationError, "coverage.exclusions"):
                    prepare_articulated_asset(asset, store=store, bake_options=BakeOptions(), support_radius=.06,
                                              excluded_links=["typo"])
                bake.assert_not_called()
                prepared = prepare_articulated_asset(asset, store=store, bake_options=BakeOptions(), support_radius=.06,
                                                      excluded_links=["auxiliary"])
                self.assertEqual(list(prepared.boundaries), [0])
                self.assertTrue(any(i.code == "coverage.excluded" for i in prepared.report.issues))


@unittest.skipUnless(wp.is_cuda_available(), "Requires actual CUDA boundary baking")
class ArticulatedBuilds(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="articulated-", dir=OUTPUT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.store = AssetStore(self.temp.name)
        self.options = BakeOptions(voxel_size=.04, padding_voxels=3, max_support_radius=.06)

    def prepare(self, asset, **kwargs):
        return prepare_articulated_asset(asset, store=self.store, bake_options=self.options,
                                         support_radius=.06, quadrature_points=3, device="cuda:0", **kwargs)

    def test_actual_newton_mesh_convex_cylinder_cold_build_reuse_and_read_only(self):
        builder = newton.ModelBuilder()
        source = trimesh.creation.box(extents=[.2, .16, .12])
        mesh = newton.Mesh(np.array(source.vertices, dtype=np.float32),
                           np.array(source.faces, dtype=np.int32).ravel())
        for kind in ("mesh", "convex", "cylinder"):
            link = builder.add_link(label=kind)
            joint = builder.add_joint_free(link)
            builder.add_articulation([joint], label=kind)
            pose = wp.transform((.03, .02, -.01), wp.quat_identity())
            if kind == "mesh":
                builder.add_shape_mesh(link, mesh=mesh, scale=(1., 1.5, .8), xform=pose, label=kind)
            elif kind == "convex":
                builder.add_shape_convex_hull(link, mesh=mesh, xform=pose, label=kind)
            else:
                builder.add_shape_cylinder(link, radius=.08, half_height=.12, xform=pose, label=kind)
        model = finalize_model(builder, device="cpu", report=ValidationReport())
        bindings = describe_articulations(model)
        self.assertEqual([b.asset.bodies[0].shapes[0].kind for b in bindings], ["MESH", "CONVEX_MESH", "CYLINDER"])
        prepared = [self.prepare(b.asset) for b in bindings]
        before = file_snapshot(self.store.root)
        self.assertEqual(len(list(self.store.root.rglob("boundary.nvdb"))), 3)
        for binding, previous in zip(bindings, prepared):
            again = self.prepare(binding.asset)
            self.assertEqual(again.boundaries, previous.boundaries)
            loaded = load_boundary_asset(again.boundaries[0], support_radius=.06, quadrature_points=3, device="cuda:0")
            self.assertGreater(loaded.sdf.get_voxel_count(), 0)
            self.assertGreater(loaded.volume_map.get_voxel_count(), 0)
        self.assertEqual(before, file_snapshot(self.store.root))
        # Body names, mass and global body indices are not part of geometry identity.
        original = bindings[0].asset
        renamed = replace(original, name="instance two", bodies=(replace(original.bodies[0], name="different", mass=900.),))
        self.assertEqual(self.prepare(renamed).boundaries, prepared[0].boundaries)
        self.assertEqual(before, file_snapshot(self.store.root))
        loaded.mesh_path.unlink()
        readonly = file_snapshot(self.store.root)
        load_boundary_asset(prepared[-1].boundaries[0], support_radius=.06, quadrature_points=3, device="cuda:0")
        self.assertEqual(readonly, file_snapshot(self.store.root))

    def test_real_g1_valid_limb_build_and_bad_mesh_report(self):
        model, bindings, _ = import_model("data/assets/g1/g1_mesh.xml")
        asset = bindings[0].asset
        good = next(body for body in asset.bodies if body.name == "left_hip_roll_link")
        result = self.prepare(PhysicalAsset("g1_valid_limb", (good,)))
        loaded = load_boundary_asset(result.boundaries[0], support_radius=.06, quadrature_points=3, device="cuda:0")
        self.assertGreater(loaded.sdf.get_voxel_count(), 0)
        before = file_snapshot(self.store.root)
        with self.assertRaisesRegex(AssetValidationError, "left_hip_pitch_link") as error:
            self.prepare(asset, excluded_links=("waist_yaw_link", "waist_roll_link"))
        self.assertTrue(any(i.code == "geometry.shape" and "triangle" in i.message for i in error.exception.report.issues))
        self.assertEqual(before, file_snapshot(self.store.root))


if __name__ == "__main__":
    unittest.main()
